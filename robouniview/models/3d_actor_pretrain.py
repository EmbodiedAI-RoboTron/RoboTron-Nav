import torch
from einops import rearrange, repeat
from torch import nn
import copy
from open_flamingo.src.helpers import PerceiverResampler
from robouniview.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder
from robouniview.models.transformers.bevformer import DeformableTransformer
from robouniview.models.transformers.position_encoding import PositionEmbeddingSine, RotaryPositionEncoding3D
from collections import namedtuple
import yaml
import argparse
import copy
import yaml
from lff_lightning.machine_learning.global_config import GlobalConfig
import numpy as np
import cv2
import pybullet as p
from robouniview.models.transformers.petr import PETR
from robouniview.models.loss_func import (
    FocalLoss, Balanced_BCE_loss, CELoss, BinaryDiceLoss, CELossIgnoreSem)

from robouniview.models.occ_head import FastEncoderHead,Upsample2d_3d,Decoder_3d
from lff_lightning.machine_learning.network_modules.losses.l1_loss import l1_loss

def load_global_config_yaml_only(config_path: str) -> GlobalConfig:
    """This function is used to init global config from yaml only, no arg parser or data version updating involved.
    It's not for job launching purpose."""
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)
    global_config = GlobalConfig(config)
    return global_config


class MPTFlamingo(nn.Module):
    def __init__(
        self,
        args,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        use_media_placement_augmentation: bool = False,
        # this is the window size sampled from the episode
        window_size: int = 8,
        use_gripper=False,
        fusion_mode='',
        sep_resampler=False,
        use_state=False,
        use_diff=False,
        diff_horizon=32,
        last_action=False,
        n_timesteps=150,
        state_dim=15,
        use_hist=False,
        debug=False,
        predict_epsilon=True,
        pad_length=-1,
        multi_step_action=1,
        sep_lm_head=False,
        return_feature = False,
        llm='llama',
        pooling='max',
        residual=False,
        tcp_rel=False,
        replan=-1,
        decoder_type='lstm',
        hidden_size=None,
        fwd_pred=False,
        fwd_pred_hand=False,
        global_latent=10,
        no_image_patch=False,
        refresh=-1
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()
        self.args = args
        self.occ_loss =  args.occ_loss
        self.train_action = args.train_action
        global_config = load_global_config_yaml_only(args.bevformer_config)
        self.uniview_feature = global_config.uniview_feature
        self.global_config = global_config
        self.fusion_mode = fusion_mode

        
        if self.fusion_mode == 'bevformer':
            if 'static' in self.uniview_feature:
                self.bevformer = DeformableTransformer(global_config,global_config.Transformer['transformer_config'])
            if 'gripper' in self.uniview_feature:
                self.bevformer_gripper = DeformableTransformer(global_config,global_config.Transformer['transformer_gripper_config'])

            if self.occ_loss:
                layers_config = {'in_channels': 1024, 'out_channels': 160, 'upsample': 2, 'head_module': 'regnet950MF'}

                self.occ_decoder = FastEncoderHead(layers_config)

                self.balanced_bce_loss = Balanced_BCE_loss(
                    1,
                    reduction="mean",
                )
        elif self.fusion_mode == 'petr':
            self.petr = PETR()

        self.position_embedding = PositionEmbeddingSine(global_config.Transformer["transformer_config"]["hidden_dim"]/2, normalize=True)
        self.use_feature = global_config.use_feature
        print('++++' * 10)
        print('++++' * 10)
        print('bevformer_config:', args.bevformer_config)
        print('use_feature:', global_config.use_feature)
        print('++++' * 10)
        print('++++' * 10)

        if self.fusion_mode == 'bevformer3D':
            if 'static' in self.uniview_feature:
                self.bevformer = DeformableTransformer(global_config,global_config.Transformer['transformer_config'])
            if 'gripper' in self.uniview_feature:
                self.bevformer_gripper = DeformableTransformer(global_config,global_config.Transformer['transformer_gripper_config'])

            self.Upsample2d_3d = Upsample2d_3d()

            self.linear = nn.Linear(128, 1024)

            self.RotaryPositionEncoding3D = RotaryPositionEncoding3D(1024)

            if self.occ_loss:
                self.MaxPool3d_4x4 = nn.AvgPool3d(4, stride=4)
            
                self.occ_decoder = Decoder_3d()

                self.balanced_bce_loss = Balanced_BCE_loss(
                    1,
                    reduction="mean",
                )
            
        self.use_gripper = use_gripper
        self.use_state = use_state
        
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.vis_dim = vis_dim
        self.window_size = window_size
        self.tcp_rel = tcp_rel
        self.act_step = multi_step_action
        print('window size: {}'.format(window_size))
        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.sep_resampler = sep_resampler
        self.use_hist = use_hist
        self.lang_encoder = lang_encoder
        self.pad_length = pad_length
        self.replan = replan
        if self.replan != -1:
            self.replan = min(int(replan * self.window_size), 180)
        self.refresh = refresh
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.residual = residual
        print(self.vis_dim, self.lang_dim)
        print(lang_encoder.config)
        if not debug:
            if 'llama' in llm:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    use_media_placement_augmentation=self.use_media_placement_augmentation,
                    residual=residual,
                )
            else:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    lang_hidden_size=self.lang_dim,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    gradient_checkpointing=False,
                )

        if sep_resampler:
            self.perceiver_gripper = PerceiverResampler(dim=self.vis_dim)
            self.perceiver_gripper.load_state_dict(copy.deepcopy(self.perceiver.state_dict()))
        if use_state:
            self.state_fc = nn.Linear(state_dim, self.vis_dim)
        if use_hist:
            self.frame_embs = nn.Parameter(torch.randn(self.window_size, self.vis_dim))
        # To-do: nn archiecture for actor
        self.llm = llm
        if llm=='llama':
            in_features = lang_encoder.lm_head.in_features
        else:
            in_features = self.lang_dim
        self.use_diff = use_diff
        self.decoder_type = decoder_type
        if decoder_type == 'lstm':
            lm_head = DeterministicDecoder(in_features, self.window_size, 
            use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling)
            self.lang_encoder.lm_head = lm_head
        elif decoder_type == 'fc':
            if use_hist:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            elif 'vit_concat' in fusion_mode:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            else:
                raise NotImplementedError
        elif decoder_type == 'diffusion':
            if use_diff:
                self.diffusion_model = DiffusionDecoder(
                    self.action_head.hidden_size, 
                    self.window_size,
                    input_dim=self.action_head.out_features+1,
                    n_timesteps=n_timesteps,
                    horizon=diff_horizon,
                    predict_epsilon=predict_epsilon,
                )
            else:
                raise NotImplementedError
        elif decoder_type=='gpt':
            lm_head = GPTDecoder(in_features, self.window_size, use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling, hidden_size=hidden_size)
            self.lang_encoder.lm_head = self.action_head = lm_head
        else:
            raise NotImplementedError
        
        sep_lm_head = True
        self.sep_lm_head = sep_lm_head
        if sep_lm_head:
            self.lm_head = self.lang_encoder.lm_head
            self.lang_encoder.lm_head = nn.Identity()
        self.env = None

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        state_tensor = None,
        calib = None,
        pcd = None,
        return_feature = False,
        policy_mask=None
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        raw_rgb = vision_x.clone()
        raw_gripper = vision_gripper.clone()
        self.pcd = pcd
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
           
            # intrinsic = calib[cam]['intrinsic_matrix'].detach().cpu().numpy()
            # extrinsic = calib[cam]['extrinsic_matrix'].detach().cpu().numpy()
            # distortion = calib[cam]['distCoeffs_matrix'].detach().cpu().numpy()
            # calib_batch['cam3']['intrinsic_matrix']
            
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.use_hist:
                self._encode_history_vision_post_fusion(vision_x, vision_gripper)
            else:
                if not self.use_gripper or self.fusion_mode == 'two_way':
                    vision_x = self._encode_vision_x(vision_x=vision_x)
                else:
                    if self.fusion_mode == 'pre':
                        self._encode_multi_vision_pre_fusion(vision_x, vision_gripper)
                    elif self.fusion_mode == 'post':
                        self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
                    elif self.fusion_mode == 'vit_concat':
                        self._encode_history_vision_fc_post(vision_x, vision_gripper)
                    elif self.fusion_mode == 'bevformer':
                        _, feats = self._encode_multi_vision_bevformer_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)
                    elif self.fusion_mode == 'bevformer3D':
                        _, feats = self._encode_multi_vision_bevformer3D_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)
                    elif self.fusion_mode == 'petr':
                        self._encode_multi_vision_petr_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)

        if self.train_action:
            output = self.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask.bool(),
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True
            )

            output_hs = output.hidden_states[-1]
            output_hs = self.lm_head(output_hs, state_tensor=state_tensor, return_feature=return_feature)
            output.logits = output_hs
        else:
            output = []
        if self.occ_loss and self.pcd is not None:
            loss_occ = self.loss_occ()
        else:
            loss_occ = {}
        
        return output,loss_occ

    


    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_vision(self, vision_x: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _encode_multi_vision_pre_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_x = torch.cat([vision_rgb, vision_gripper], dim=3)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_rgb = self.perceiver(vision_rgb)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_two_way(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_rgb = self.perceiver(vision_rgb)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=0)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=0)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_history_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert False
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        bs = int(vision_rgb.shape[0] // self.window_size)
        vision_rgb = vision_rgb.view(bs, self.window_size, *vision_rgb.shape[1:])
        _, _, T, p, v_tok, dim = vision_rgb.shape[:6]
        frame_embs = repeat(self.frame_embs, 'F d -> b F T p v d', b=bs, T=T, p=p, v=v_tok)
        vision_rgb = vision_rgb + frame_embs
        vision_rgb = rearrange(vision_rgb, 'b F T p v d -> (b F) T p v d')
        vision_rgb = self.perceiver(vision_rgb)

        vision_gripper = vision_gripper.view(vision_gripper.shape[0] // self.window_size, self.window_size,
                                             *vision_gripper.shape[1:])
        frame_embs = repeat(self.frame_embs, 'F d -> b F T p v d', b=bs, T=T, p=p, v=v_tok)
        vision_gripper = vision_gripper + frame_embs
        vision_gripper = rearrange(vision_gripper, 'b F T p v d -> (b F) T p v d')
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x
    
    def _encode_history_vision_fc_post(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert False
        bs = int(vision_rgb.shape[0] // self.window_size)
        vision_rgb = self._encode_vision(vision_rgb)
        vision_rgb = self.perceiver(vision_rgb) # BxL, T, n, d
        vision_rgb = vision_rgb.view(-1, self.window_size, *vision_rgb.shape[1:]) # B, L, T, n, d
        vision_rgb = rearrange(vision_rgb, 'b L T n d -> b T (n L) d')

        vision_gripper = self._encode_vision(vision_gripper)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)
        vision_gripper = vision_gripper.view(-1, self.window_size, *vision_gripper.shape[1:]) # B, L, T, n, d
        vision_gripper = rearrange(vision_gripper, 'b L T n d -> b T (n L) d')

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)

        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x
    
    
    def _encode_multi_vision_bevformer_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, calib ,state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        # vision_rgb_img = vision_rgb[0][0][0]
        # vision_rgb_img = rearrange(vision_rgb_img, "c h w  -> h w c")
        # vision_rgb_img = np.array(vision_rgb_img.cpu())
        # cv2.imwrite("~/liufanfan/workspace/RoboFlamingo/rgb_ft.jpg", vision_rgb_img*200)


        # vision_gripper_img = vision_gripper[0][0][0]
        # vision_gripper_img = rearrange(vision_gripper_img, "c h w  -> h w c")
        # vision_gripper_img = np.array(vision_gripper_img.cpu())
        # cv2.imwrite("~/liufanfan/workspace/RoboFlamingo/rgb_gripper_ft.jpg", vision_gripper_img*200)

        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)


        feats = {}

        if 'rgb_static' in self.use_feature:
            vision_rgb_1d = vision_rgb.clone()
            vision_rgb_1d = self.perceiver(vision_rgb_1d)
            feats['rgb_static'] = vision_rgb_1d

        if 'rgb_gripper' in self.use_feature:
            vision_gripper_1d = vision_gripper.clone()
            if self.sep_resampler:
                vision_gripper_1d = self.perceiver_gripper(vision_gripper_1d)
            else:
                vision_gripper_1d = self.perceiver(vision_gripper_1d)
            feats['rgb_gripper'] = vision_gripper_1d

        if 'bev' in self.use_feature:
            
            B, T, F, HxW, C = vision_rgb.shape
            #vision_rgb = vision_rgb.reshape(B*T, C, H, W)
            vision_rgb = rearrange(vision_rgb, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
            vision_gripper = rearrange(vision_gripper, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
            
            _calib1 = {'rgb_static':{'extrinsic_matrix':rearrange(calib[0]," B T H W -> (B T) H W").cpu(),
                                    'intrinsic_matrix':rearrange(calib[1]," B T H W -> (B T) H W").cpu(),
                                    'distCoeffs_matrix':rearrange(calib[2]," B T H -> (B T) H").cpu()},
                        'rgb_gripper':{'extrinsic_matrix':rearrange(calib[3]," B T H W -> (B T) H W").cpu(),
                                        'intrinsic_matrix':rearrange(calib[4]," B T H W -> (B T) H W").cpu(),
                                        'distCoeffs_matrix':rearrange(calib[5]," B T H -> (B T) H").cpu()}}
            x = [[vision_rgb],[vision_gripper]]

            if 'static' in self.uniview_feature:

                bev_feat = self.bevformer(x, _calib1) #(B*T, C, H, W)
                occ_feat = bev_feat.clone()


                if self.occ_loss :
                    self.occ = self.occ_decoder(occ_feat)
                    self.occ = rearrange(self.occ, "BT (Z C) H W -> BT H W Z C",Z=40,C=4)


                pos = self.position_embedding(bev_feat)
                bev_feat = bev_feat + pos
                bev_feat = rearrange(bev_feat, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)
                feats['bev_static'] = bev_feat

            if 'gripper' in self.uniview_feature:
                
                rg_em = _calib1['rgb_gripper']['extrinsic_matrix'].clone()
                rs_em = _calib1['rgb_static']['extrinsic_matrix'].clone()
                for i, _rg_em in enumerate(rg_em):
                    _calib1['rgb_gripper']['extrinsic_matrix'][i]  = rg_em[i] @ torch.linalg.inv(rg_em[i])
                for i, _rs_em in enumerate(rs_em):
                    _calib1['rgb_static']['extrinsic_matrix'][i]  = rs_em[i] @ torch.linalg.inv(rg_em[i])
                bev_feat = self.bevformer_gripper(x, _calib1) #(B*T, C, H, W)
                pos = self.position_embedding(bev_feat)
                bev_feat = bev_feat + pos
                bev_feat = rearrange(bev_feat, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)
                feats['bev_gripper'] = bev_feat

               
        vision_x = []
        for feats_key in feats.keys():
            vision_x.append(feats[feats_key])
        
        vision_x = torch.concatenate(vision_x, dim=2)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)


       
            
        return vision_x, feats
    

    def loss_occ(self):
        """
        Args:
            self.preds: shape of (bs, w, h, z, c)
            self.trues: shape of (bs, w, h, z, c)
        """
        # parse preds

        # self.occ = np.array(self.occ)
        # self.occ_true = np.array(self.occ_true)

        self.occ_true = self.pcd
        self.occ_true = rearrange(self.occ_true, "B T H W Z C  -> (B T) H W Z C")

        c_classes = self.occ_true.shape[-1]
        grid_cls = ['occ','r','g','b']
        loss ={}
        self.loss_weight_dict = [0.5,0.25,0.25,0.25]
        for ind in range(c_classes):
            preds_ind = self.occ[:,:,:,:, ind]
            trues_ind = self.occ_true[:,:,:,:, ind] 
            if ind == 0: 
                loss_ind = self.balanced_bce_loss(preds_ind, trues_ind)
            else:
                #loss_ind = ((preds_ind-trues_ind)*trues[:,:,:,:, 0]).mean()

                loss_ind = l1_loss(preds_ind, trues_ind, self.occ_true[:,:,:,:, 0])
            loss[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * self.loss_weight_dict[ind]
        return loss

    
    def _encode_multi_vision_bevformer3D_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, calib ,state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        # vision_rgb_img = vision_rgb[0][0][0]
        # vision_rgb_img = rearrange(vision_rgb_img, "c h w  -> h w c")
        # vision_rgb_img = np.array(vision_rgb_img.cpu())
        # cv2.imwrite("~/liufanfan/workspace/RoboFlamingo/rgb_ft.jpg", vision_rgb_img*200)


        # vision_gripper_img = vision_gripper[0][0][0]
        # vision_gripper_img = rearrange(vision_gripper_img, "c h w  -> h w c")
        # vision_gripper_img = np.array(vision_gripper_img.cpu())
        # cv2.imwrite("~/liufanfan/workspace/RoboFlamingo/rgb_gripper_ft.jpg", vision_gripper_img*200)

        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)


        feats = {}

        if 'rgb_static' in self.use_feature:
            vision_rgb_1d = vision_rgb.clone()
            vision_rgb_1d = self.perceiver(vision_rgb_1d)
            feats['rgb_static'] = vision_rgb_1d

        if 'rgb_gripper' in self.use_feature:
            vision_gripper_1d = vision_gripper.clone()
            if self.sep_resampler:
                vision_gripper_1d = self.perceiver_gripper(vision_gripper_1d)
            else:
                vision_gripper_1d = self.perceiver(vision_gripper_1d)
            feats['rgb_gripper'] = vision_gripper_1d

        if 'bev' in self.use_feature:
            
            B, T, F, HxW, C = vision_rgb.shape
            #vision_rgb = vision_rgb.reshape(B*T, C, H, W)
            vision_rgb = rearrange(vision_rgb, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
            vision_gripper = rearrange(vision_gripper, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
            
            _calib1 = {'rgb_static':{'extrinsic_matrix':rearrange(calib[0]," B T H W -> (B T) H W").cpu(),
                                    'intrinsic_matrix':rearrange(calib[1]," B T H W -> (B T) H W").cpu(),
                                    'distCoeffs_matrix':rearrange(calib[2]," B T H -> (B T) H").cpu()},
                        'rgb_gripper':{'extrinsic_matrix':rearrange(calib[3]," B T H W -> (B T) H W").cpu(),
                                        'intrinsic_matrix':rearrange(calib[4]," B T H W -> (B T) H W").cpu(),
                                        'distCoeffs_matrix':rearrange(calib[5]," B T H -> (B T) H").cpu()}}
            x = [[vision_rgb],[vision_gripper]]

            if 'static' in self.uniview_feature:
                bev_feat = self.bevformer(x, _calib1) #(B*T, C, H, W)
                occ_feat1 , occ_feat2 = self.Upsample2d_3d(bev_feat)#(B*T, C, Z, H, W)
                occ_feat = occ_feat1.clone()
                feats_pcd = self.generate_all_feats_pcd(occ_feat1)#(B*T, Z, H, W, 3)
                feats_pcd = feats_pcd.to(occ_feat1.device)
                if self.occ_loss :
                    self.occ = self.occ_decoder(occ_feat)
                    occ_confidence = self.occ.clone()
                    occ_confidence[:,0,:,:,:] = torch.sigmoid(occ_confidence[:,0,:,:,:])
                    occ_confidence = self.MaxPool3d_4x4(occ_confidence)
                    self.occ = rearrange(self.occ, "BT C Z H W -> BT H W Z C")
                    occ_confidence = rearrange(occ_confidence, "BT C Z H W -> BT H W Z C")
                    feats_pcd = rearrange(feats_pcd, "BT Z H W C -> BT H W Z C")
                    occ_feat1 = rearrange(occ_feat1, "BT C Z H W -> BT H W Z C")
                    occ_confidence = rearrange(occ_confidence, "BT H W Z C -> BT (H W Z) C") #(B*T,L,C)
                    feats_pcd = rearrange(feats_pcd, "BT H W Z C -> BT (H W Z) C") #(B*T,L,C)
                    occ_feat1 = rearrange(occ_feat1, "BT H W Z C -> BT (H W Z) C") #(B*T,L,C)
                    #occ_confidence[:, :, 0] = occ_confidence[:, :, 0].sigmoid
                    values = occ_confidence[:, :, 0]
                    sorted_indices = torch.argsort(values, dim=1, descending=True)
                    occ_confidence = torch.gather(occ_confidence, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, occ_confidence.size(2)))
                    occ_confidence = occ_confidence[:, :400, :]
                    feats_pcd = torch.gather(feats_pcd, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, feats_pcd.size(2)))
                    feats_pcd = feats_pcd[:, :400, :]
                    occ_feat1 = torch.gather(occ_feat1, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, occ_feat1.size(2)))
                    occ_feat1 = occ_feat1[:, :400, :]
                    occ_feat1 = self.linear(occ_feat1)
                    _relative = True
                    
                    if _relative:
                        center = rearrange(state_tensor, " B T F C  -> (B T F) C")[:, :3]
                        bs = center.shape[0]
                        feats_pcd = feats_pcd - center.view(bs, 1, 3)

                    pos_min = torch.tensor([[-0.4,-0.4,0.4]]).float().to(feats_pcd.device)
                    pos_max = torch.tensor([[0.4,0.4,0.7]]).float().to(feats_pcd.device)
                    feats_pcd = (feats_pcd - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
                    feats_pcd = self.RotaryPositionEncoding3D(feats_pcd)

                # pos = self.position_embedding(bev_feat)
                # bev_feat = bev_feat + pos
                # bev_feat = rearrange(bev_feat, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)

                occ_feat1 = occ_feat1+feats_pcd[:,:,0:occ_feat1.shape[2]]
                occ_feat1 = rearrange(occ_feat1, " (B T) L C ->B T L C", B=B, T=T)
                feats['bev_static'] = occ_feat1

            if 'gripper' in self.uniview_feature:
                
                rg_em = _calib1['rgb_gripper']['extrinsic_matrix'].clone()
                rs_em = _calib1['rgb_static']['extrinsic_matrix'].clone()
                for i, _rg_em in enumerate(rg_em):
                    _calib1['rgb_gripper']['extrinsic_matrix'][i]  = rg_em[i] @ torch.linalg.inv(rg_em[i])
                for i, _rs_em in enumerate(rs_em):
                    _calib1['rgb_static']['extrinsic_matrix'][i]  = rs_em[i] @ torch.linalg.inv(rg_em[i])
                bev_feat = self.bevformer_gripper(x, _calib1) #(B*T, C, H, W)
                pos = self.position_embedding(bev_feat)
                bev_feat = bev_feat + pos
                bev_feat = rearrange(bev_feat, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)
                feats['bev_gripper'] = bev_feat

               
        vision_x = []
        for feats_key in feats.keys():
            vision_x.append(feats[feats_key])
        
        vision_x = torch.concatenate(vision_x, dim=2)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        # if self.occ_loss :
        #     self.occ = self.occ_decoder(occ_feat)
        #     self.occ = rearrange(self.occ, "BT (Z C) H W -> BT H W Z C",Z=40,C=4)
            
        return vision_x, feats
    
    def generate_all_feats_pcd(self, x):
        b, c, z, h, w = x.shape
        scalex = 1 / w
        scaley = 1 / h
        scalez = 0.5 / z

        grid_x = torch.linspace(-0.5 + scalex/2, 0.5 - scalex/2, steps=w)
        grid_y = torch.linspace(-0.5 + scaley/2, 0.5 - scaley/2, steps=h)
        grid_z = torch.linspace(0.3 + scalez/2, 0.8 - scalez/2, steps=z)

        mesh_z , mesh_x, mesh_y = torch.meshgrid(grid_z, grid_x, grid_y, indexing='ij')
        index_3D = torch.stack((mesh_x, mesh_y, mesh_z), dim=-1)
        index_3D = index_3D.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        index_3D = rearrange(index_3D, "B H W Z C -> B Z H W C")

        return index_3D
        
                    





    
    def _encode_multi_vision_petr_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)

        vision_rgb = rearrange(vision_rgb, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
        vision_gripper = rearrange(vision_gripper, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
        state_tensor = rearrange(state_tensor, " B T F C  -> (B T F) C")
        x = [[vision_rgb],[vision_gripper]]
        calib = {'rgb_static':{'intrinsic_matrix':{},'extrinsic_matrix':{},'distCoeffs_matrix':{}},
                'rgb_gripper':{'intrinsic_matrix':{},'extrinsic_matrix':{},'distCoeffs_matrix':{}}}
        calib['rgb_static']['extrinsic_matrix'] = look2extrinsic(self.global_config.camera['rgb_static']['extrinsic']['look_at'], 
                                                            self.global_config.camera['rgb_static']['extrinsic']['look_from'],
                                                            self.global_config.camera['rgb_static']['extrinsic']['up_vector'])
        calib['rgb_static']['extrinsic_matrix'] = torch.from_numpy(np.array(calib['rgb_static']['extrinsic_matrix'])).unsqueeze(0).repeat(vision_rgb.shape[0],1,1)
        calib['rgb_static']['intrinsic_matrix'] = torch.from_numpy(np.array(self.global_config.camera['rgb_static']['intrinsic'])).unsqueeze(0).repeat(vision_rgb.shape[0],1,1)   
        calib['rgb_static']['distCoeffs_matrix'] = torch.from_numpy(np.array(self.global_config.camera['rgb_static']['distCoeffs'])).repeat(vision_rgb.shape[0],1)
        calib['rgb_gripper']['intrinsic_matrix'] = torch.from_numpy(np.array(self.global_config.camera['rgb_gripper']['intrinsic'])).unsqueeze(0).repeat(vision_rgb.shape[0],1,1)
        calib['rgb_gripper']['distCoeffs_matrix'] = torch.from_numpy(np.array(self.global_config.camera['rgb_gripper']['distCoeffs'])).repeat(vision_rgb.shape[0],1)
        calib['rgb_gripper']['extrinsic_matrix'] = calib['rgb_static']['extrinsic_matrix'].clone()
        state = state_tensor[:,:6].cpu()
        state_mat = pose_vec2mat(state)
        for i, _state_mat in enumerate(state_mat):
                calib['rgb_gripper']['extrinsic_matrix'][i]  = \
                torch.linalg.inv(pose_vec2mat(torch.tensor([-state[0][1] , state[0][0] ,state[0][2], -state[0][3], state[0][4],-state[0][5]]))) \
                @ pose_vec2mat(torch.tensor(self.global_config.camera['rgb_gripper']['extrinsic']))

        img_mates = [] 
        if 'static' in self.uniview_feature:
            
            for i ,_extrinsic_matrix in enumerate(calib['rgb_static']['extrinsic_matrix']):
                lidar2img = []
                _intrinsic_matrix = calib['rgb_static']['intrinsic_matrix'][i]
                _extrinsic_matrix = np.array(_extrinsic_matrix)
                _intrinsic_matrix = np.array(_intrinsic_matrix)
                lidar2img.append((_intrinsic_matrix @ _extrinsic_matrix.T))
                _extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix'][i]
                _intrinsic_matrix = calib['rgb_gripper']['intrinsic_matrix'][i]
                _extrinsic_matrix = np.array(_extrinsic_matrix)
                _intrinsic_matrix = np.array(_intrinsic_matrix)
                lidar2img.append((_intrinsic_matrix @ _extrinsic_matrix.T))
                img_mate = {'lidar2img':lidar2img}
                img_mates.append(img_mate)

        lidar2imgs = [] 
        if 'gripper' in self.uniview_feature:
            
            rg_em = calib['rgb_gripper']['extrinsic_matrix'].clone()
            rs_em = calib['rgb_static']['extrinsic_matrix'].clone()
            for i, _state_mat in enumerate(state_mat):
                calib['rgb_gripper']['extrinsic_matrix'][i]  = rg_em[i] @ torch.linalg.inv(rg_em[i])
            for i, _state_mat in enumerate(state_mat):
                calib['rgb_static']['extrinsic_matrix'][i]  = rs_em[i] @ torch.linalg.inv(rg_em[i])

            for i ,_extrinsic_matrix in enumerate(calib['rgb_static']['extrinsic_matrix']):
                lidar2img = []
                _intrinsic_matrix = calib['rgb_static']['intrinsic_matrix'][i]
                _extrinsic_matrix = np.array(_extrinsic_matrix)
                _intrinsic_matrix = np.array(_intrinsic_matrix)
                lidar2img.append((_intrinsic_matrix @ _extrinsic_matrix.T))

                _extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix'][i]
                _intrinsic_matrix = calib['rgb_gripper']['intrinsic_matrix'][i]
                _extrinsic_matrix = np.array(_extrinsic_matrix)
                _intrinsic_matrix = np.array(_intrinsic_matrix)

                lidar2img.append((_intrinsic_matrix @ _extrinsic_matrix.T))
            lidar2imgs.append(lidar2img)



        vision_rgb = self.perceiver(vision_rgb)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x
    
def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    if vec is None:
        return None
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat

def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def look2extrinsic(look_at,look_from,up_vector):
    look_at = np.array(look_at)
    look_from = np.array(look_from)
    up_vector = np.array(up_vector)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = np.array(look_from)
    #cam z
    cam_pose[:3, 2] = np.array(look_at) - cam_pose[:3, 3]
    cam_pose[:3, 2] /= np.linalg.norm(cam_pose[:3, 2])
    #cam y
    y_axis = -np.array(up_vector)
    cam_pose[:3, 1] = y_axis - np.sum(y_axis * cam_pose[:3, 2]) * cam_pose[:3, 2]   # 正交化
    cam_pose[:3, 1] /= np.linalg.norm(cam_pose[:3, 1])
    #cam x
    cam_pose[:3, 0] = np.cross(cam_pose[:3, 1], cam_pose[:3, 2])
    cam_pose = np.linalg.inv(cam_pose)
    return cam_pose


def scale_intrinsic(f_ratio, intrinsic):
    intrinsic[0:2][:] *= f_ratio
    return intrinsic