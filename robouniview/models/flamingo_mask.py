import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import *
from einops import rearrange, repeat
from transformers import AutoModelForCausalLM


def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    query_mask_matrix: Optional[torch.ByteTensor] = None
):

    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=n_heads)  # includes key.t()
    v = rearrange(value, 'b s (h d) -> b h s d', h=n_heads)

    min_val = torch.finfo(q.dtype).min

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if attn_bias is not None:
        if (attn_bias.size(-1) != 1 and
                attn_bias.size(-1) != s_k) or (attn_bias.size(-2) != 1 and
                                               attn_bias.size(-2) != s_q):
            raise RuntimeError(
                f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.'
            )
        attn_weight = attn_weight + attn_bias

    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn(
                'Propogating key_padding_mask to the attention module ' +\
                'and applying it within the attention module can cause ' +\
                'unneccessary computation/memory usage. Consider integrating ' +\
                'into attn_bias once and passing that to each attention ' +\
                'module instead.'
            )
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if query_mask_matrix is not None:
        assert len(query_mask_matrix.shape)==3 and len(attn_weight.shape)==4 and query_mask_matrix.shape[0] == attn_weight.shape[0] and query_mask_matrix.shape[1] == attn_weight.shape[2] and query_mask_matrix.shape[2] == attn_weight.shape[3]
        attn_weight = attn_weight.masked_fill(~query_mask_matrix.view((b, 1, s_k, s_k)), min_val)

    if is_causal:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k),
                                              min_val)

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight,
                                                  p=dropout_p,
                                                  training=training,
                                                  inplace=True)

    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')

    if needs_weights:
        return out, attn_weight
    return out, None


class MultiheadAttentionMaskMixin(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available():
                warnings.warn(
                    'Using `attn_impl: torch`. If your model does not use `alibi` or ' +\
                    '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' +\
                    'we recommend using `attn_impl: triton`.'
                )

    def forward(self,
                x,
                past_key_value=None,
                attn_bias=None,
                attention_mask=None,
                is_causal=True,
                needs_weights=False,
                query_mask_matrix: Optional[torch.ByteTensor] = None):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.chunk(3, dim=2)

        key_padding_mask = attention_mask

        if self.attn_qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        if past_key_value is not None:
            if len(past_key_value) != 0:
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)

            past_key_value = (key, value)

        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]

        if self.attn_impl == 'torch': # 不使用self.attn_fn
            context, attn_weights = scaled_multihead_dot_product_attention(
                query,
                key,
                value,
                self.n_heads,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                dropout_p=self.attn_dropout_p,
                training=self.training,
                needs_weights=needs_weights,
                query_mask_matrix=query_mask_matrix,
            )

        return self.out_proj(context), attn_weights, past_key_value


class GPTBlockMaskMixin(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.extend_instance(self.attn, MultiheadAttentionMaskMixin)

    def extend_instance(self, instance, mixin):
        instance.__class__ = type(
            instance.__class__.__name__,
            (mixin, instance.__class__),
            {}
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        query_mask_matrix: Optional[torch.ByteTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        a = self.ln_1(x)
        b, _, past_key_value = self.attn(a,
                                         past_key_value=past_key_value,
                                         attn_bias=attn_bias,
                                         attention_mask=attention_mask,
                                         is_causal=is_causal,
                                         query_mask_matrix=query_mask_matrix)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        return x, past_key_value


class FlamingoLMMaskMixin(nn.Module):
    """
    Mixin to add self-attention layers for mask matrix to a language model.
    """
    
    def forward_super(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            query_mask_matrix: Optional[torch.ByteTensor] = None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MosaicGPT
        if not return_dict:
            raise NotImplementedError(
                'return_dict False is not implemented yet for MosaicGPT')
        if output_attentions:
            raise NotImplementedError(
                'output_attentions is not implemented yet for MosaicGPT')

        if attention_mask is not None and attention_mask[:, 0].sum(
        ) != attention_mask.shape[0] and self.training:
            raise NotImplementedError(
                'MosaicGPT does not support training with left padding.')

        if self.prefix_lm and prefix_mask is None:
            raise ValueError(
                'prefix_mask is a required argument when MosaicGPT is configured with prefix_lm=True.'
            )

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when MosaicGPT is configured with attn_uses_sequence_id=True ' +\
                    'and the model is in train mode.'
                )
            elif (self.attn_uses_sequence_id is False) and (sequence_id
                                                            is not None):
                warnings.warn(
                    'MosaicGPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' +\
                    'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.'
                )

        S = input_ids.size(1)

        assert (
            S <= self.config.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        if self.alibi:
            x = tok_emb # 默认没有添加pos编码
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        f'past_key_values must provide a past_key_value for each attention ' +\
                        f'layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).'
                    )
                # get the key tensor whose spec should be (batch, seq, dim), and
                # collect the `seq`, so that the position embedding is shifted
                past_position = past_key_values[0][0].size(1)

            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length '
                    f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.'
                )
            pos = torch.arange(past_position,
                               S + past_position,
                               dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            if attention_mask is not None:
                # adjust the position indices to account for padding tokens
                pos = torch.clamp(pos - torch.cumsum(
                    (~attention_mask).to(torch.int32), dim=1)[:,
                                                              past_position:],
                                  min=0)

            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = tok_emb + pos_emb

        if self.embedding_fraction == 1:
            x = self.transformer.emb_drop(x)  # type: ignore 使用drop mask掉部分输入，但drop rate目前为0
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.transformer.emb_drop, nn.Module)  # pyright
            x = self.transformer.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=x.dtype,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id)

        # initialize the past key values cache if it should be used
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)
                              ]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        for b_idx, block in enumerate(self.transformer.blocks):  # type: ignore
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,) # x表示语言
            past_key_value = past_key_values[
                b_idx] if past_key_values is not None else None
            x, past_key_value = block(x,
                                      past_key_value=past_key_value,
                                      attn_bias=attn_bias,
                                      attention_mask=attention_mask,
                                      is_causal=self.is_causal,
                                      query_mask_matrix=query_mask_matrix)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value

        x = self.transformer.ln_f(x)  # type: ignore

        # output embedding weight tied to input embedding
        assert isinstance(self.transformer.wte, nn.Module)  # pyright
        assert isinstance(self.transformer.wte.weight, torch.Tensor)  # pyright
        logits = F.linear(x, self.transformer.wte.weight, None)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        return CausalLMOutputWithPast(logits=logits,
                                      past_key_values=past_key_values,
                                      hidden_states=all_hidden_states)

class FlamingoLMAutoModelForCausalLM(FlamingoLMMaskMixin, AutoModelForCausalLM):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        
        # 遍历transformer.blocks并对子模块进行扩展
        for block in self.model.transformer.h:
            self.extend_block(block)

    def extend_block(self, block):
        # 扩展模块
        self.extend_instance(block, GPTBlockMaskMixin)
    
    def extend_instance(self, instance, mixin):
        instance.__class__ = type(
            instance.__class__.__name__,
            (mixin, instance.__class__),
            {}
        )
