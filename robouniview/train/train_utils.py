import time
from contextlib import suppress

import torch, gc
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from robouniview.utils import world_to_tcp_frame, tcp_to_world_frame
import itertools
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from robouniview.models.loss_func import (
    FocalLoss, Balanced_BCE_loss, CELoss, BinaryDiceLoss, CELossIgnoreSem, l1_loss,MSE_Loss)
from collections import defaultdict

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
    

def get_ckpt_name(args, epoch=-1):
    if args.use_gripper:
        ckpt_name = 'checkpoint_gripper_{}_hist_{}_{}'.format(args.fusion_mode, args.hist_window, '' if not args.sep_resampler else 'sep_')
    else:
        ckpt_name = 'checkpoint_no_gripper_hist_{}_{}'.format(args.hist_window, '' if not args.sep_resampler else 'sep_')
    if args.real_data:
        ckpt_name += 'real_'
    if args.train_params != -1:
        ckpt_name += 'train_{}_'.format(args.train_params)
    if args.no_pretrain:
        ckpt_name += 'no_pretrain_'
    if args.fwd_pred:
        ckpt_name += 'pred_rgb_'
    if args.fwd_pred_hand:
        ckpt_name += 'pred_hand_'
    if args.freeze_sampler:
        ckpt_name += 'freeze_sam_'
    if args.use_state:
        ckpt_name += 'state_'
    if args.rgb_pad != -1 or args.gripper_pad != -1:
        ckpt_name += 'aug_{}_{}_'.format(args.rgb_pad, args.gripper_pad)
    if args.use_hist:
        ckpt_name += 'fc_'
    if args.head_type == "diffusion":
        ckpt_name += 'diff_'
    if args.traj_cons:
        ckpt_name += 'traj_cons_'
    if args.sep_lm_head:
        ckpt_name += 'lm_head_'
    if args.dif_ws:
        ckpt_name += 'difws_{}_{}_'.format(args.min_window_size, args.max_window_size)
    elif args.window_size != 8:
        ckpt_name += 'ws_{}_'.format(args.window_size)
    if args.unfreeze_vit:
        ckpt_name += 'unfreeze_vit_'
    if args.llm_name != 'llama':
        ckpt_name += '{}_'.format(args.llm_name)
    if args.pooling != 'max':
        ckpt_name += '{}_'.format(args.pooling)
    if args.text_aug:
        ckpt_name += 'text_aug_'
    if args.residual:
        ckpt_name += 'res_'
    if args.freeze_embed:
        ckpt_name += 'freeze_emb_'
    if args.tcp_rel:
        ckpt_name += 'tcp_'
    if args.multi_step_action != 1:
        ckpt_name += '{}_fur_step_'.format(args.multi_step_action)
    if args.decoder_type != 'lstm':
        ckpt_name += '{}_{}_'.format(args.decoder_type, args.hidden_size)
    if args.lr_scheduler != 'constant':
        ckpt_name += '{}_'.format(args.lr_scheduler)
    ckpt_name += '{}.pth'.format(epoch)

    if epoch != -1:
        if epoch >= 100:
            ckpt_name += '{}_iter.pth'.format(epoch)
        else:
            ckpt_name += '{}.pth'.format(epoch)
    else:
        ckpt_name += 'final_weights.pth'
    return ckpt_name

def get_ckpt_name_pattern(args):
    if args.use_gripper:
        ckpt_name = 'checkpoint_gripper_{}_hist_{}_{}'.format(args.fusion_mode, args.hist_window, '' if not args.sep_resampler else 'sep_')
    else:
        ckpt_name = 'checkpoint_no_gripper_hist_{}_{}'.format(args.hist_window, '' if not args.sep_resampler else 'sep_')
    if args.real_data:
        ckpt_name += 'real_'
    if args.train_params != -1:
        ckpt_name += 'train_{}_'.format(args.train_params)
    if args.no_pretrain:
        ckpt_name += 'no_pretrain_'
    if args.fwd_pred:
        ckpt_name += 'pred_rgb_'
    if args.fwd_pred_hand:
        ckpt_name += 'pred_hand_'
    if args.freeze_sampler:
        ckpt_name += 'freeze_sam_'
    if args.use_state:
        ckpt_name += 'state_'
    if args.rgb_pad != -1 or args.gripper_pad != -1:
        ckpt_name += 'aug_{}_{}_'.format(args.rgb_pad, args.gripper_pad)
    if args.use_hist:
        ckpt_name += 'fc_'
    if args.head_type == "diffusion":
        ckpt_name += 'diff_'
    if args.traj_cons:
        ckpt_name += 'traj_cons_'
    if args.sep_lm_head:
        ckpt_name += 'lm_head_'
    if args.dif_ws:
        ckpt_name += 'difws_{}_{}_'.format(args.min_window_size, args.max_window_size)
    elif args.window_size != 8:
        ckpt_name += 'ws_{}_'.format(args.window_size)
    if args.unfreeze_vit:
        ckpt_name += 'unfreeze_vit_'
    if args.llm_name != 'llama':
        ckpt_name += '{}_'.format(args.llm_name)
    if args.pooling != 'max':
        ckpt_name += '{}_'.format(args.pooling)
    if args.text_aug:
        ckpt_name += 'text_aug_'
    if args.residual:
        ckpt_name += 'res_'
    if args.freeze_embed:
        ckpt_name += 'freeze_emb_'
    if args.tcp_rel:
        ckpt_name += 'tcp_'
    if args.multi_step_action != 1:
        ckpt_name += '{}_fur_step_'.format(args.multi_step_action)
    if args.decoder_type != 'lstm':
        ckpt_name += '{}_{}_'.format(args.decoder_type, args.hidden_size)
    if args.lr_scheduler != 'constant':
            ckpt_name += '{}_'.format(args.lr_scheduler)
    ckpt_name += '*.pth'
    return ckpt_name

def train_one_epoch_calvin_diff(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    normalizer=None,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    if isinstance(model, DistributedDataParallel):
        diffusion_model = model.module.diffusion_model
    else:
        diffusion_model = model.diffusion_model
    
    if normalizer is None:
        normalizer = diffusion_model.normalizer

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []

    if isinstance(model, DistributedDataParallel):
        action_dim = model.module.action_head.out_features + 1 # joint + gripper
    else:
        action_dim = model.action_head.out_features + 1 # joint + gripper
 
    class LowdimMaskGenerator(nn.Module):
        def __init__(self,
            action_dim, obs_dim,
            # obs mask setup
            max_n_obs_steps=3, 
            fix_obs_steps=True, 
            # action mask
            action_visible=True,
            return_one_mask=False
            ):
            super().__init__()
            self.action_dim = action_dim
            self.obs_dim = obs_dim
            self.max_n_obs_steps = max_n_obs_steps
            self.fix_obs_steps = fix_obs_steps
            self.action_visible = action_visible
            self.return_one_mask = return_one_mask

        @torch.no_grad()
        def forward(self, shape, device, seed=None):
            # device = self.device
            B, T, D = shape
            assert D == (self.action_dim + self.obs_dim)

            # create all tensors on this device
            rng = torch.Generator(device=device)
            if seed is not None:
                rng = rng.manual_seed(seed)

            # generate dim mask
            dim_mask = torch.zeros(size=shape, 
                dtype=torch.bool, device=device)
            is_action_dim = dim_mask.clone()
            is_action_dim[...,:self.action_dim] = True
            is_obs_dim = ~is_action_dim

            # generate obs mask
            if self.fix_obs_steps:
                obs_steps = torch.full((B,), 
                fill_value=self.max_n_obs_steps, device=device)
            else:
                obs_steps = torch.randint(
                    low=1, high=self.max_n_obs_steps+1, 
                    size=(B,), generator=rng, device=device)
                
            steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
            obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
            obs_mask = obs_mask

            # generate action mask
            if self.action_visible:
                action_steps = torch.maximum(
                    obs_steps - 1, 
                    torch.tensor(0,
                        dtype=obs_steps.dtype, 
                        device=obs_steps.device))
                action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
                action_mask = action_mask & is_action_dim


            if self.return_one_mask:
                mask = obs_mask & is_obs_dim
                if self.action_visible:
                    mask = mask | action_mask
            
                return mask
            if self.obs_dim <= 0:
                assert self.fix_obs_steps, "We require fix obs steps to obtain obs masks"
                obs_mask = obs_mask[0,:,0]
            return action_mask, obs_mask     

    mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=args.n_obs_steps,
            fix_obs_steps=True,
            action_visible=True,
    )

    act_mask, obs_mask = None, None
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = batch_calvin[1][0].unsqueeze(1).repeat(1, images.shape[1], 1)

        # do the same to the attention mask 
        attention_mask = batch_calvin[1][1].unsqueeze(1).repeat(1, images.shape[1], 1)
        state_tensor = batch_calvin[4].unsqueeze(2).unsqueeze(2)

        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        actions = normalizer.normalize(actions) # labels normalization

        if act_mask is None or obs_mask is None:
            act_mask, obs_mask = mask_generator(actions.shape, images.device)

        batch_size = actions.shape[0]
        # Mask and leave history data for generating features
        images = images[:,obs_mask,...]
        gripper = gripper[:,obs_mask,...]
        input_ids = input_ids[:,obs_mask,...]
        attention_mask = attention_mask[:,obs_mask,...]
        state_tensor = state_tensor[:,obs_mask,...]

         # put images and labels on device
        images = images.to(device_id, dtype=cast_dtype, non_blocking=True)
        gripper = gripper.to(device_id, dtype=cast_dtype, non_blocking=True)

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = input_ids.to(device_id, non_blocking=True)

        # do the same to the attention mask 
        attention_mask = attention_mask.to(device_id, non_blocking=True)
        state_tensor = state_tensor.to(device_id, dtype=cast_dtype, non_blocking=True)

        # print("test", images.shape, gripper.shape, input_ids.shape, attention_mask.shape, state_tensor.shape)
        # import pdb; pdb.set_trace()
        
        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        input_ids = input_ids.flatten(0, 1)
        attention_mask = attention_mask.flatten(0, 1)

        with autocast():
            model_out = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            ) # Features
            model_out = model_out.logits

        # compute loss
        tt = torch.randint(0, args.n_timesteps, (batch_size,), device=actions.device).long()
        noise = torch.randn_like(actions)
        
        action_noisy = diffusion_model.q_sample(x_start=actions, t=tt, noise=noise)
 
        # apply conditioning
        action_noisy[act_mask] = actions[act_mask]
        # pred = diffusion_model(action_noisy, tt, global_cond=None)
        pred = diffusion_model(action_noisy, tt, global_cond=model_out)
        pred[act_mask] = actions[act_mask] # So we remove the gradient
        assert noise.shape == pred.shape

        if args.predict_epsilon:
            loss = F.mse_loss(pred, noise, reduction='none')
        else:
            loss = F.mse_loss(pred, actions, reduction='none')

        loss_calvin = loss.mean()

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item()})


def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    writer,
    evaluate_func=None,
    start_from_iter=0,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    calvin_loader.sampler.set_num_skip_iters(start_from_iter * calvin_loader.batch_size) # dataloader中更新iter

    autocast = get_autocast(args.precision) # FP32
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    action_token_id = tokenizer("<action>", add_special_tokens=False)["input_ids"][-1]
    static_token_ids, gripper_token_ids, obs_token_ids = [], [], []
    for i_th in range(8):
        static_token_ids.append(tokenizer(f"<static{i_th}>", add_special_tokens=False)["input_ids"][-1])
        gripper_token_ids.append(tokenizer(f"<gripper{i_th}>", add_special_tokens=False)["input_ids"][-1])
        obs_token_ids.append(tokenizer(f"<obs{i_th}>", add_special_tokens=False)["input_ids"][-1])
    static_token_ids = torch.tensor(static_token_ids).to(device_id, non_blocking=True)
    gripper_token_ids = torch.tensor(gripper_token_ids).to(device_id, non_blocking=True)
    obs_token_ids = torch.tensor(obs_token_ids).to(device_id, non_blocking=True)

    model.train()
    # setup logging
    step_time_m = (AverageMeter())  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (AverageMeter())  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch + start_from_iter), # 进度条中更新iter
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    calvin_avg_loss = []
    occ_avg_loss = []
    
    _l1_loss = torch.nn.L1Loss(reduction="none")
    balanced_bce_loss = Balanced_BCE_loss(1, reduction="mean",)
    ed_yf = time.time(); TESTTIME=False
    print(f"start from iter {epoch * num_batches_per_epoch + start_from_iter} on all epochs") # 输出中更新iter
    for num_steps, batch_calvin in t:
        
        if TESTTIME and torch.distributed.get_rank() == 0: st_yf = time.time(); print(f"YF: data loder: {st_yf-ed_yf}")
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch + start_from_iter # 保存ckpt的时候，需要从all epochs的start from iter继续计算

        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))     

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        #if args.fusion_mode != 'vit_concat':
        if 'Temporal' not in args.fusion_mode:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1) # input toekn重复12次
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1).bool() # attention_mask重复12次
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).bool()
            action_mask = torch.any(input_ids[..., None] == action_token_id, dim=2)
            static_mask = torch.any(input_ids[..., None] == static_token_ids, dim=2) # bool型
            gripper_mask = torch.any(input_ids[..., None] == gripper_token_ids, dim=2)
            obs_mask = torch.any(input_ids[..., None] == obs_token_ids, dim=2)
            if not static_mask.any(): static_mask = None
            if not gripper_mask.any(): gripper_mask = None
            if not obs_mask.any(): obs_mask = None

        state_tensor = batch_calvin[4] # .to(device_id, dtype=cast_dtype, non_blocking=True)
        robot_obs = batch_calvin[5] # .to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)
        state_tensor = state_tensor.flatten(0, 1)

        calib = batch_calvin[6] # YF: 0.image;1.text;2.action;3.gripper; 4.state;5(static extr;static intr;static_dist...;static fov),6:pcd;7:index
        pcd = batch_calvin[7].to(device_id, dtype=cast_dtype, non_blocking=True)
        idxes_sample = batch_calvin[8]
        # print(idxes_sample)
        # merge the batch and the sequence dimension
        # images = images.flatten(0, 1)
        # gripper = gripper.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist: labels = labels[:, [-1]]  # only calculate last step action
        if args.fusion_mode == 'vit_concat': labels = labels[:, -1]
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]

        if TESTTIME and torch.distributed.get_rank() == 0: ed_pd_yf = time.time(); print(f"YF: data prepress: {ed_pd_yf-st_yf} {idxes_sample}")
        with autocast():
            output, static_pred, gripper_pred, obs_pred, aux_loss = model(
                vision_x=images[:, :args.window_size],
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper[:, :args.window_size],
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                calib = calib, # 注意里边实际上有args.window_size+1个
                pcd = pcd[:, :args.window_size], # UVFormer使用
                action_mask = action_mask,
                static_mask = static_mask,
                gripper_mask = gripper_mask,
                obs_mask = obs_mask
            )
        # np.save("tmp.npy", {"images": images[:,:12].cpu().numpy(), "input_ids": input_ids.cpu().numpy(), "attention_mask": attention_mask.cpu().numpy(), "gripper": gripper[:,:12].cpu().numpy(), "state_tensor": state_tensor.cpu().numpy() if (args.use_state or args.sep_lm_head) else None, "calib": calib, "action_mask": action_mask.cpu().numpy(), "static_mask": static_mask, "gripper_mask": gripper_mask, "obs_mask":obs_mask,"pcd":pcd.cpu().numpy(), "output": output, "static_pred": static_pred, "gripper_pred": gripper_pred})
        if TESTTIME and torch.distributed.get_rank() == 0: ed_fd_yf = time.time(); print(f"YF: model forward: {ed_fd_yf-ed_pd_yf}")
        loss_rgb, loss_gripper_rgb = torch.tensor(0), torch.tensor(0)
        if static_mask is not None :
            assert len(images[0]) == args.window_size + 1
            raw_rgb = rearrange(images[:, 1:], "B T F G D H W -> (B T F G) D H W")
            raw_rgb = F.interpolate(raw_rgb, (112, 112), mode='bilinear')
            rgbmask = torch.ones_like(raw_rgb)
            loss_rgb = MSE_Loss(static_pred, raw_rgb, rgbmask) * 0.1
        if gripper_mask is not None :
            raw_gripper = rearrange(gripper[:,1:], "B T F G D H W -> (B T F G) D H W")
            raw_gripper = F.interpolate(raw_gripper, (112, 112), mode='bilinear')
            grippermask = torch.ones_like(raw_gripper)
            loss_gripper_rgb = MSE_Loss(gripper_pred, raw_gripper, grippermask) * 0.1
            #loss_gripper_rgb = loss_gripper_rgb.sum() / (loss_gripper_rgb.shape[0] * loss_gripper_rgb.shape[1] * loss_gripper_rgb.shape[2] * loss_gripper_rgb.shape[3])

        loss_obs = defaultdict(float)
        if obs_mask is not None :
            pcd = rearrange(pcd[:, :args.window_size], "B T H W Z C  -> (B T) H W Z C")
            c_classes = pcd.shape[-1]
            grid_cls = ['occ','r','g','b']
            for ind in range(c_classes):
                preds_ind = obs_pred[:,:,:,:, ind]
                trues_ind = pcd[:,:,:,:, ind] 
                if ind == 0: 
                    loss_ind = balanced_bce_loss(preds_ind, trues_ind)
                else:
                    loss_ind = MSE_Loss(preds_ind, trues_ind, pcd[:,:,:,:, 0])
                loss_obs[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * args.occ_loss_weight[ind]
        if args.occ_loss and 'loss_occ' in aux_loss:
            loss_occ = aux_loss['loss_occ']
            for ind, k in enumerate(loss_occ.keys()):
                loss_obs[f"grid_cls_{k}_loss"] += loss_occ[k] * args.occ_loss_weight[ind]

        # compute loss
        if args.multi_action_token:
            num_actions, bin_actions = output.logits[0], output.logits[1]
            # reshape for loss calculation
            if args.multi_step_action != 1:
                assert False
                bs, seq_len = num_actions.shape[:2]
                num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            if 'Temporal' in args.fusion_mode:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][:, :args.window_size, :])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1][:, :args.window_size, :])
            else:
                assert False
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        elif args.train_action:
            assert False
            num_actions, bin_actions = output.logits[0], output.logits[1]
            # reshape for loss calculation
            if args.multi_step_action != 1:
                bs, seq_len = num_actions.shape[:2]
                num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            if 'Temporal' in args.fusion_mode:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][:,-1,:])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1][:,-1,:])
            else:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        else:
            loss_calvin_num = torch.tensor(0).to(device_id)
            loss_calvin_bin = torch.tensor(0).to(device_id)
        if args.real_data:
            loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        else:
            loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps
        #### BACKWARD PASS ####
        loss = (divided_loss_calvin * args.loss_multiplier_calvin)
        loss = loss * args.loss_weight['action']
        for k in loss_obs.keys():
            loss += loss_obs[k]*args.loss_weight['occ'] # YF: wight在这里乘的的，同时没有使用oRGB的wight
        loss = loss + loss_rgb + loss_gripper_rgb

        mv_avg_loss.append(loss.item())
        calvin_avg_loss.append(loss_calvin.item())
        if 'grid_cls_occ_loss' in loss_obs:
            occ_avg_loss.append(loss_obs['grid_cls_occ_loss'].item())

        loss.backward()
        if TESTTIME and torch.distributed.get_rank() == 0: ed_bk_yf = time.time(); print(f"YF: backword: {ed_bk_yf - ed_fd_yf}")
        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
                m.weight.grad = m.weight.grad * zero_mask
        # model.apply(mask_embedding)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        writer.add_scalar("totoal_norm",total_norm, num_steps)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (num_steps == num_batches_per_epoch - 1):

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], num_steps)
            writer.add_scalar("loss", loss.item(), num_steps)
            writer.add_scalar("loss_action", divided_loss_calvin.item(), num_steps)
            for k in loss_obs.keys():
                writer.add_scalar(k,loss_obs[k].item(), num_steps)
        if TESTTIME and torch.distributed.get_rank() == 0: ed_up_yf = time.time(); print(f"YF: update: {ed_up_yf - ed_bk_yf}")
        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}")
        loss_dic = {"avg_action_loss": sum(calvin_avg_loss[-min(100,len(calvin_avg_loss)):]) / min(100,len(calvin_avg_loss)), "action_loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item(), "avg_occ_loss": sum(occ_avg_loss[-min(100,len(occ_avg_loss)):]) / (min(100,len(occ_avg_loss))+0.00001),}
        for k in loss_obs.keys():
            loss_dic[k] = loss_obs[k].item()
        loss_dic['lr'] = optimizer.param_groups[0]["lr"]
        # loss_rgb + loss_gripper_rgb
        loss_dic['loss_rgb'] = loss_rgb.item()
        loss_dic['loss_gripper_rgb'] = loss_gripper_rgb.item()
        t.set_postfix(loss_dic)

        if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
            if args.rank == 0:
                import os
                if not os.path.exists(args.run_name): os.makedirs(args.run_name)
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "cur_iter": num_steps + start_from_iter, # 保存ckpt的时候，需要cur_iter从current epoch的iter继续计算
                }
                ckpt_name = get_ckpt_name(args, global_step)
                ckpt_path = os.path.join(args.run_name, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint_dict, ckpt_path)
                if args.delete_previous_checkpoint: 
                    if epoch > 0: os.remove(ckpt_path)
        # if global_step % 1000 == 0 and args.rank == 0 and global_step > 0:
        #     import os
        #     if not os.path.exists(args.run_name): os.makedirs(args.run_name)
        #     checkpoint_dict = {
        #         "epoch": epoch,
        #         "model_state_dict": get_checkpoint(model),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #         "cur_iter": num_steps,
        #     }
        #     ckpt_name = get_ckpt_name(args, 1001)
        #     ckpt_path = os.path.join(args.run_name, ckpt_name)
        #     print(f"Saving checkpoint to {ckpt_path}")
        #     torch.save(checkpoint_dict, ckpt_path)
        
        # gc.collect()
        # torch.cuda.empty_cache()
        if args.eval and evaluate_func is not None and args.eval_steps>0 and global_step % args.eval_steps == 0 and global_step > 0:
            evaluate_func(model)
        if TESTTIME and torch.distributed.get_rank() == 0: ed_yf = time.time(); print(f"YF: one iter: {ed_yf - st_yf}")
        # if global_step > 10: return


def train_one_epoch_move(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    writer,
    evaluate_func=None,
    start_from_iter=0,
):
    # 计算iter数
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    calvin_loader.sampler.set_num_skip_iters(start_from_iter * calvin_loader.batch_size) # dataloader中更新iter
    # 设置精度
    autocast = get_autocast(args.precision) # FP32
    cast_dtype = get_cast_dtype(args.precision)
    # 计算特殊token id
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1] # 将字符"<image>"变成数字50278
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1] # 将字符"<|endofchunk|>"变成数字50277
    action_token_id = tokenizer("<action>", add_special_tokens=False)["input_ids"][-1] # 将字符"<action>"变成数字50279
    static_token_ids, gripper_token_ids, obs_token_ids = [], [], []
    for i_th in range(8):
        static_token_ids.append(tokenizer(f"<static{i_th}>", add_special_tokens=False)["input_ids"][-1]) # 50280-50287
        gripper_token_ids.append(tokenizer(f"<gripper{i_th}>", add_special_tokens=False)["input_ids"][-1]) # 50290-50297
        obs_token_ids.append(tokenizer(f"<obs{i_th}>", add_special_tokens=False)["input_ids"][-1]) # 50300-50307
    static_token_ids = torch.tensor(static_token_ids).to(device_id, non_blocking=True) # torch.Size([8])
    gripper_token_ids = torch.tensor(gripper_token_ids).to(device_id, non_blocking=True) # torch.Size([8])
    obs_token_ids = torch.tensor(obs_token_ids).to(device_id, non_blocking=True) # torch.Size([8])

    model.train()
    # setup logging
    step_time_m = (AverageMeter())  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (AverageMeter())  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch + start_from_iter), # 进度条中更新iter
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = [] # 总loss，即机械臂loss+点云loss
    calvin_avg_loss = [] # 机械臂loss
    occ_avg_loss = [] # 点云loss中的occ loss
    
    _l1_loss = torch.nn.L1Loss(reduction="none")
    balanced_bce_loss = Balanced_BCE_loss(1, reduction="mean",)
    ed_yf = time.time(); TESTTIME=False
    print(f"start from iter {epoch * num_batches_per_epoch + start_from_iter} on all epochs") # 输出中更新iter
    for num_steps, batch_calvin in t:
        
        if TESTTIME and torch.distributed.get_rank() == 0: st_yf = time.time(); print(f"YF: data loder: {st_yf-ed_yf}")
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch + start_from_iter # 保存ckpt的时候，需要从all epochs的start from iter继续计算

        # put images and labels on device 为什么要做这一步？？？
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)) # torch.Size([4, 13, 1, 1, 3, 224, 224])
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)) # torch.Size([4, 13, 1, 1, 3, 224, 224])

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        #if args.fusion_mode != 'vit_concat':
        if 'Temporal' not in args.fusion_mode:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1) # input toekn重复12次
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1).bool() # attention_mask重复12次
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True) # torch.Size([4, 146])
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).bool() # torch.Size([4, 146])
            action_mask = torch.any(input_ids[..., None] == action_token_id, dim=2) # torch.Size([4, 146])
            static_mask = torch.any(input_ids[..., None] == static_token_ids, dim=2) # bool型
            gripper_mask = torch.any(input_ids[..., None] == gripper_token_ids, dim=2)
            obs_mask = torch.any(input_ids[..., None] == obs_token_ids, dim=2)
            if not static_mask.any(): static_mask = None # none
            if not gripper_mask.any(): gripper_mask = None # none
            if not obs_mask.any(): obs_mask = None # none

        state_tensor = batch_calvin[4] # .to(device_id, dtype=cast_dtype, non_blocking=True) torch.Size([4, 13, 15])
        robot_obs = batch_calvin[5] # .to(device_id, dtype=cast_dtype, non_blocking=True) torch.Size([1])
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True) # torch.Size([4, 13, 7])
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2) # torch.Size([4, 13, 1, 1, 15]) 为什么要进行unsqueeze(2).unsqueeze(2)的操作？
        state_tensor = state_tensor.flatten(0, 1) # torch.Size([52, 1, 1, 15])

        calib = batch_calvin[6] # zyf: 0.sem; 1.sim; 2.sdm; 3.gem; 4.gim; 5.gtm; 6.sm; 7.sf(fhw); 8.gf(fhw)
        pcd = batch_calvin[7].to(device_id, dtype=cast_dtype, non_blocking=True) # torch.Size([4, 13, 80, 80, 40, 4])
        idxes_sample = batch_calvin[8] # torch.Size([4])
        his_vision_static = batch_calvin[9] # 4 12 20 1024
        his_pose = batch_calvin[10] # 4 12 20 3
        # print(idxes_sample)
        # merge the batch and the sequence dimension
        # images = images.flatten(0, 1)
        # gripper = gripper.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist: labels = labels[:, [-1]]  # only calculate last step action
        if args.fusion_mode == 'vit_concat': labels = labels[:, -1]
        labels = [labels[..., :6], labels[..., 6:]] # torch.Size([4, 13, 6]) torch.Size([4, 13, 1])

        if TESTTIME and torch.distributed.get_rank() == 0: ed_pd_yf = time.time(); print(f"YF: data prepress: {ed_pd_yf-st_yf} {idxes_sample}")

        with autocast():
            output, static_pred, gripper_pred, obs_pred, aux_loss = model(
                vision_x=images[:, :args.window_size],
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper[:, :args.window_size],
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                calib = calib, # 注意里边实际上有args.window_size+1个
                pcd = pcd[:, :args.window_size], # UVFormer使用
                action_mask = action_mask,
                static_mask = static_mask,
                gripper_mask = gripper_mask,
                obs_mask = obs_mask,
                his_vision_static = his_vision_static,
                his_pose = his_pose,
            )
     
        # np.save("tmp.npy", {"images": images[:,:12].cpu().numpy(), "input_ids": input_ids.cpu().numpy(), "attention_mask": attention_mask.cpu().numpy(), "gripper": gripper[:,:12].cpu().numpy(), "state_tensor": state_tensor.cpu().numpy() if (args.use_state or args.sep_lm_head) else None, "calib": calib, "action_mask": action_mask.cpu().numpy(), "static_mask": static_mask, "gripper_mask": gripper_mask, "obs_mask":obs_mask,"pcd":pcd.cpu().numpy(), "output": output, "static_pred": static_pred, "gripper_pred": gripper_pred})
        if TESTTIME and torch.distributed.get_rank() == 0: ed_fd_yf = time.time(); print(f"YF: model forward: {ed_fd_yf-ed_pd_yf}")
        loss_rgb, loss_gripper_rgb = torch.tensor(0), torch.tensor(0)
        if static_mask is not None :
            assert len(images[0]) == args.window_size + 1
            raw_rgb = rearrange(images[:, 1:], "B T F G D H W -> (B T F G) D H W")
            raw_rgb = F.interpolate(raw_rgb, (112, 112), mode='bilinear')
            rgbmask = torch.ones_like(raw_rgb)
            loss_rgb = MSE_Loss(static_pred, raw_rgb, rgbmask) * 0.1
        if gripper_mask is not None :
            raw_gripper = rearrange(gripper[:,1:], "B T F G D H W -> (B T F G) D H W")
            raw_gripper = F.interpolate(raw_gripper, (112, 112), mode='bilinear')
            grippermask = torch.ones_like(raw_gripper)
            loss_gripper_rgb = MSE_Loss(gripper_pred, raw_gripper, grippermask) * 0.1
            #loss_gripper_rgb = loss_gripper_rgb.sum() / (loss_gripper_rgb.shape[0] * loss_gripper_rgb.shape[1] * loss_gripper_rgb.shape[2] * loss_gripper_rgb.shape[3])

        loss_obs = defaultdict(float)
        if obs_mask is not None :
            pcd = rearrange(pcd[:, :args.window_size], "B T H W Z C  -> (B T) H W Z C")
            c_classes = pcd.shape[-1]
            grid_cls = ['occ','r','g','b']
            for ind in range(c_classes):
                preds_ind = obs_pred[:,:,:,:, ind]
                trues_ind = pcd[:,:,:,:, ind] 
                if ind == 0: 
                    loss_ind = balanced_bce_loss(preds_ind, trues_ind)
                else:
                    loss_ind = MSE_Loss(preds_ind, trues_ind, pcd[:,:,:,:, 0])
                loss_obs[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * args.occ_loss_weight[ind]
        if args.occ_loss and 'loss_occ' in aux_loss:
            loss_occ = aux_loss['loss_occ']
            for ind, k in enumerate(loss_occ.keys()):
                loss_obs[f"grid_cls_{k}_loss"] += loss_occ[k] * args.occ_loss_weight[ind]

        # compute loss
        if args.multi_action_token:
            num_actions, bin_actions = output.logits[0], output.logits[1]
            # reshape for loss calculation
            if args.multi_step_action != 1:
                assert False
                bs, seq_len = num_actions.shape[:2]
                num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            if 'Temporal' in args.fusion_mode:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][:, :args.window_size, :])
                num_classes = bin_actions.shape[-1] # bs t 20
                # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, torch.nn.functional.one_hot(labels[1][:, :args.window_size, 0].to(torch.int64), num_classes=num_classes).to(torch.float32)) # label onehot以后也是bs t 20
                # 调整预测张量的形状为 [bs * t, 20]，调整真实标签张量的形状为 [bs * t]
                loss_calvin_bin = torch.nn.functional.cross_entropy(bin_actions.view(-1, num_classes), labels[1][:, :args.window_size, 0].view(-1).to(torch.int64))
            else:
                assert False
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        elif args.train_action:
            assert False
            num_actions, bin_actions = output.logits[0], output.logits[1]
            # reshape for loss calculation
            if args.multi_step_action != 1:
                bs, seq_len = num_actions.shape[:2]
                num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            if 'Temporal' in args.fusion_mode:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][:,-1,:])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1][:,-1,:])
            else:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        else:
            loss_calvin_num = torch.tensor(0).to(device_id)
            loss_calvin_bin = torch.tensor(0).to(device_id)
        # if args.real_data:
        #     loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        # else:
        #     loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
        loss_calvin = loss_calvin_num + loss_calvin_bin

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps
        #### BACKWARD PASS ####
        loss = (divided_loss_calvin * args.loss_multiplier_calvin)
        loss = loss * args.loss_weight['action']
        for k in loss_obs.keys():
            loss += loss_obs[k]*args.loss_weight['occ'] # YF: wight在这里乘的的，同时没有使用oRGB的wight
        loss = loss + loss_rgb + loss_gripper_rgb

        mv_avg_loss.append(loss.item()) # 总loss，即机械臂loss+点云loss，后面没有使用
        calvin_avg_loss.append(loss_calvin.item()) # 机械臂loss
        if 'grid_cls_occ_loss' in loss_obs:
            occ_avg_loss.append(loss_obs['grid_cls_occ_loss'].item()) # 点云loss中的occ loss

        loss.backward()
        if TESTTIME and torch.distributed.get_rank() == 0: ed_bk_yf = time.time(); print(f"YF: backword: {ed_bk_yf - ed_fd_yf}")
        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
                m.weight.grad = m.weight.grad * zero_mask
        # model.apply(mask_embedding)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 模型中所有参数梯度的L2范数的总和会被裁剪到不超过1.0
        writer.add_scalar("totoal_norm",total_norm, num_steps)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (num_steps == num_batches_per_epoch - 1):

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], num_steps)
            writer.add_scalar("loss", loss.item(), num_steps)
            writer.add_scalar("loss_action", divided_loss_calvin.item(), num_steps)
            for k in loss_obs.keys():
                writer.add_scalar(k,loss_obs[k].item(), num_steps)
        if TESTTIME and torch.distributed.get_rank() == 0: ed_up_yf = time.time(); print(f"YF: update: {ed_up_yf - ed_bk_yf}")
        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}")
        # 计算最近100个或更少的损失值的平均值
        loss_dic = {"avg_action_loss": sum(calvin_avg_loss[-min(100,len(calvin_avg_loss)):]) / min(100,len(calvin_avg_loss)), "action_loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item(), "avg_occ_loss": sum(occ_avg_loss[-min(100,len(occ_avg_loss)):]) / (min(100,len(occ_avg_loss))+0.00001),}
        for k in loss_obs.keys():
            loss_dic[k] = loss_obs[k].item()
        loss_dic['lr'] = optimizer.param_groups[0]["lr"]
        loss_dic['succ'] = f"{(bin_actions.argmax(-1) == labels[1][:, :args.window_size, 0]).to(torch.float32).mean():.2f}"
        # loss_rgb + loss_gripper_rgb
        loss_dic['loss_rgb'] = loss_rgb.item()
        loss_dic['loss_gripper_rgb'] = loss_gripper_rgb.item()
        t.set_postfix(loss_dic)

        if not args.debug and args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
            if args.rank == 0:
                import os
                if not os.path.exists(args.run_name): os.makedirs(args.run_name)

                if args.delete_previous_checkpoint: 
                    import glob
                    ckpt_name = get_ckpt_name_pattern(args)
                    checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
                    # checkpoint_list = [_ for _ in checkpoint_list if "__sep" not in _ and 'iter' not in _ and 'weights' not in _]
                    if len(checkpoint_list) > 0:
                        print('args.delete_previous_checkpoint:', args.delete_previous_checkpoint)
                        for ckpt_path in sorted(checkpoint_list, key=os.path.getmtime)[:-3]: os.remove(ckpt_path)
                
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "cur_iter": num_steps + start_from_iter, # 保存ckpt的时候，需要cur_iter从current epoch的iter继续计算
                }
                ckpt_name = get_ckpt_name(args, global_step)
                ckpt_path = os.path.join(args.run_name, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint_dict, ckpt_path)
        # if global_step % 1000 == 0 and args.rank == 0 and global_step > 0:
        #     import os
        #     if not os.path.exists(args.run_name): os.makedirs(args.run_name)
        #     checkpoint_dict = {
        #         "epoch": epoch,
        #         "model_state_dict": get_checkpoint(model),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #     }
        #     ckpt_name = get_ckpt_name(args, 1001)
        #     ckpt_path = os.path.join(args.run_name, ckpt_name)
        #     print(f"Saving checkpoint to {ckpt_path}")
        #     torch.save(checkpoint_dict, ckpt_path)
        
        # gc.collect()
        # torch.cuda.empty_cache()
        if args.eval and evaluate_func is not None and args.eval_steps>0 and global_step % args.eval_steps == 0 and global_step > 0:
            evaluate_func(model)
        if TESTTIME and torch.distributed.get_rank() == 0: ed_yf = time.time(); print(f"YF: one iter: {ed_yf - st_yf}")
        # if global_step > 10: return


def train_one_epoch_nav_vqa(
    args,
    model,
    epoch,
    calvin_loader,
    vqa_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    writer,
    evaluate_func=None,
    start_from_iter=0,
    vqa_start_from_iter=0,
    vqa_dataset=None,
):
    # 计算iter数
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    calvin_loader.sampler.set_num_skip_iters(start_from_iter * calvin_loader.batch_size) # dataloader中更新iter
    vqa_loader.sampler.set_num_skip_iters(vqa_start_from_iter * vqa_loader.batch_size) # dataloader中更新iter

    # 设置精度
    autocast = get_autocast(args.precision) # FP32
    cast_dtype = get_cast_dtype(args.precision)
    # 计算特殊token id
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1] # 将字符"<image>"变成数字50278
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1] # 将字符"<|endofchunk|>"变成数字50277
    action_token_id = tokenizer("<action>", add_special_tokens=False)["input_ids"][-1] # 将字符"<action>"变成数字50279
    endoftxt_token_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1] # 将字符"<|endoftxt|>"变成数字0
    answer_token_id = tokenizer(" Answer", add_special_tokens=False)["input_ids"][-1] # 将字符" Answer"变成数字37741
    colon_token_id = tokenizer(":", add_special_tokens=False)["input_ids"][-1] # 将字符":"变成数字27

    static_token_ids, gripper_token_ids, obs_token_ids = [], [], []
    for i_th in range(8):
        static_token_ids.append(tokenizer(f"<static{i_th}>", add_special_tokens=False)["input_ids"][-1]) # 50280-50287
        gripper_token_ids.append(tokenizer(f"<gripper{i_th}>", add_special_tokens=False)["input_ids"][-1]) # 50290-50297
        obs_token_ids.append(tokenizer(f"<obs{i_th}>", add_special_tokens=False)["input_ids"][-1]) # 50300-50307
    static_token_ids = torch.tensor(static_token_ids).to(device_id, non_blocking=True) # torch.Size([8])
    gripper_token_ids = torch.tensor(gripper_token_ids).to(device_id, non_blocking=True) # torch.Size([8])
    obs_token_ids = torch.tensor(obs_token_ids).to(device_id, non_blocking=True) # torch.Size([8])

    def calculate_vl_cross_entropy(logits, labels, mask=None):
        shift_logits = logits[..., :-1, :].contiguous() # 2, 836, 50311 为什么这么处理？？？
        shift_labels = labels[..., 1:].contiguous() # 2, 836
        # Flatten the tokens
        if mask is None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(
                    -1, logits.shape[-1]
                ),
                shift_labels.view(-1),
            )
        else:
            # TODO: mask is with the same shape of labels, 
            # 1 represents valid, 0 for non-valid, only calculate loss for valid tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
            shift_logits.view(
                    -1, logits.shape[-1]
                ), # Flatten 
            shift_labels.view(-1), # Flatten 
            )
            # mask the loss
            mask = mask[..., 1:].contiguous()
            loss = loss * mask.reshape(-1) # 20, 128
            # mean
            loss = loss.mean()
        return loss

    model.train()
    # setup logging
    step_time_m = (AverageMeter())  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (AverageMeter())  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # 创建vqa_loader的迭代器
    vqa_iter = iter(vqa_loader)

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch + start_from_iter), # 进度条中更新iter
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = [] # 总loss，即机械臂loss+点云loss
    calvin_avg_loss = [] # 机械臂loss
    occ_avg_loss = [] # 点云loss中的occ loss
    mv_avg_loss_vqa = [] # vqa的loss

    _l1_loss = torch.nn.L1Loss(reduction="none")
    balanced_bce_loss = Balanced_BCE_loss(1, reduction="mean",)
    ed_yf = time.time(); TESTTIME=False
    print(f"start from iter {epoch * num_batches_per_epoch + start_from_iter} on all epochs") # 输出中更新iter
    for num_steps, batch_calvin in t:
        
        if TESTTIME and torch.distributed.get_rank() == 0: st_yf = time.time(); print(f"YF: data loder: {st_yf-ed_yf}")
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch + start_from_iter # 保存ckpt的时候，需要从all epochs的start from iter继续计算

        #### nav-vqa FORWARD PASS ####
        # batch_vqa = next(iter(vqa_loader))
        try:
            batch_vqa = next(vqa_iter)
        except StopIteration:
            # 如果vqa_loader迭代器耗尽，重新创建迭代器
            num_batches_per_epoch_vqa = vqa_dataset.dataloader.num_batches
            vqa_epoch = vqa_dataset.shared_epoch.get_value() + 1 # global_step // num_batches_per_epoch_vqa
            vqa_start_from_iter = 0 # global_step % num_batches_per_epoch_vqa

            vqa_dataset.set_epoch(vqa_epoch)
            vqa_loader = vqa_dataset.dataloader
            vqa_loader.sampler.set_num_skip_iters(vqa_start_from_iter * vqa_loader.batch_size) # dataloader中更新iter

            vqa_iter = iter(vqa_loader)
            batch_vqa = next(vqa_iter)
            
        images = (batch_vqa[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)) # torch.Size([2, 1, 1, 1, 3, 224, 224])
        gripper = (batch_vqa[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)) # torch.Size([2, 1, 1, 1, 3, 224, 224])

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        #if args.fusion_mode != 'vit_concat':
        if 'Temporal' not in args.fusion_mode:
            input_ids = batch_vqa[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1) # input toekn重复12次
            attention_mask = batch_vqa[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1).bool() # attention_mask重复12次
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)
        else:
            input_ids = batch_vqa[1][0].to(device_id, non_blocking=True) # torch.Size([2, 830])
            attention_mask = batch_vqa[1][1].to(device_id, non_blocking=True).bool() # torch.Size([2, 830])
            action_mask = torch.any(input_ids[..., None] == action_token_id, dim=2) # torch.Size([2, 830])
            static_mask = torch.any(input_ids[..., None] == static_token_ids, dim=2) # torch.Size([2, 830])
            gripper_mask = torch.any(input_ids[..., None] == gripper_token_ids, dim=2) # torch.Size([2, 830])
            obs_mask = torch.any(input_ids[..., None] == obs_token_ids, dim=2) # torch.Size([2, 830])
            if not static_mask.any(): static_mask = None # none
            if not gripper_mask.any(): gripper_mask = None # none
            if not obs_mask.any(): obs_mask = None # none

        # 初始化掩码为全零
        ques_mask = torch.zeros_like(input_ids, dtype=torch.int)

        # 遍历每个序列
        for i in range(input_ids.size(0)):
            # 找到连续的 (37741, 27) 的位置
            start_positions = ((input_ids[i, :-1] == answer_token_id) & (input_ids[i, 1:] == colon_token_id)).nonzero(as_tuple=True)[0]
            
            # 找到连续的 (50277, 0) 的位置
            end_positions = ((input_ids[i, :-1] == endofchunk_token_id) & (input_ids[i, 1:] == endoftxt_token_id)).nonzero(as_tuple=True)[0]
            
            # 如果找到有效的开始和结束位置
            if start_positions.numel() > 0 and end_positions.numel() > 0:
                start_pos = start_positions[0].item() + 1  # +1 to start after 27
                end_pos = end_positions[0].item()  # end before 50277
                if start_pos < end_pos:
                    ques_mask[i, start_pos:end_pos] = 1
            else: 
                print("No answer mask!!!")

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100 # 50310
        labels[labels == media_token_id] = -100 # 50278
        labels = labels.to(device_id) # 20 129

        state_tensor = batch_vqa[4] # .to(device_id, dtype=cast_dtype, non_blocking=True) torch.Size([2, 1, 15])
        robot_obs = batch_vqa[5] # .to(device_id, dtype=cast_dtype, non_blocking=True) torch.Size([1])

        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)

        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2) # torch.Size([2, 1, 1, 15])
        state_tensor = state_tensor.flatten(0, 1) # torch.Size([2, 1, 1, 15])

        calib = batch_vqa[6] # zyf: 0.sem; 1.sim; 2.sdm; 3.gem; 4.gim; 5.gtm; 6.sm; 7.sf(fhw); 8.gf(fhw)
        pcd = batch_vqa[7].to(device_id, dtype=cast_dtype, non_blocking=True) # torch.Size([2, 1, 80, 80, 40, 4])
        idxes_sample = batch_vqa[8] # torch.Size([4])
        his_vision_static = batch_vqa[9] # torch.Size([2, 1, 60, 1024])
        his_pose = batch_vqa[10] # torch.Size([2, 1, 60, 3])

        if TESTTIME and torch.distributed.get_rank() == 0: ed_pd_yf = time.time(); print(f"YF: data prepress: {ed_pd_yf-st_yf} {idxes_sample}")

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        assert args.vqa_use_static == True, "In vqa dataset, args.vqa_use_static must be True!!!"
        with autocast():
            output = model(
                vision_x=images[:, :args.window_size],
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper[:, :args.window_size],
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                calib = calib, # 注意里边实际上有args.window_size+1个
                pcd = pcd[:, :args.window_size], # UVFormer使用
                action_mask = action_mask,
                static_mask = static_mask,
                gripper_mask = gripper_mask,
                obs_mask = obs_mask,
                his_vision_static = his_vision_static,
                his_pose = his_pose,
                use_vqa = True,
                use_static = args.vqa_use_static, # 只有vqa调用flamingo的时候，use_static为true
            )
        
        logits = output.logits # 20, 129, 50311
        loss_vqa = calculate_vl_cross_entropy(logits, labels, ques_mask)
        mv_avg_loss_vqa.append(loss_vqa.item())
        divided_loss_vqa = loss_vqa 
        divided_loss_vqa = divided_loss_vqa / args.gradient_accumulation_steps
        # (divided_loss_vqa * args.loss_multiplier_calvin).backward() # 为什么不加在一起反向传播？
        loss_vqa = divided_loss_vqa * args.loss_multiplier_calvin

        #### action FORWARD PASS ####
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)) # torch.Size([4, 12, 1, 1, 3, 224, 224])
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)) # torch.Size([4, 12, 1, 1, 3, 224, 224])

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        #if args.fusion_mode != 'vit_concat':
        if 'Temporal' not in args.fusion_mode:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1) # input toekn重复12次
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1).bool() # attention_mask重复12次
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True) # torch.Size([4, 146])
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).bool() # torch.Size([4, 146])
            action_mask = torch.any(input_ids[..., None] == action_token_id, dim=2) # torch.Size([4, 146])
            static_mask = torch.any(input_ids[..., None] == static_token_ids, dim=2) # bool型
            gripper_mask = torch.any(input_ids[..., None] == gripper_token_ids, dim=2)
            obs_mask = torch.any(input_ids[..., None] == obs_token_ids, dim=2)
            if not static_mask.any(): static_mask = None # none
            if not gripper_mask.any(): gripper_mask = None # none
            if not obs_mask.any(): obs_mask = None # none

        state_tensor = batch_calvin[4] # .to(device_id, dtype=cast_dtype, non_blocking=True) torch.Size([4, 13, 15])
        robot_obs = batch_calvin[5] # .to(device_id, dtype=cast_dtype, non_blocking=True) torch.Size([1])
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True) # torch.Size([4, 13, 7])
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2) # torch.Size([4, 13, 1, 1, 15]) 为什么要进行unsqueeze(2).unsqueeze(2)的操作？
        state_tensor = state_tensor.flatten(0, 1) # torch.Size([52, 1, 1, 15])

        calib = batch_calvin[6] # zyf: 0.sem; 1.sim; 2.sdm; 3.gem; 4.gim; 5.gtm; 6.sm; 7.sf(fhw); 8.gf(fhw)
        pcd = batch_calvin[7].to(device_id, dtype=cast_dtype, non_blocking=True) # torch.Size([4, 13, 80, 80, 40, 4])
        idxes_sample = batch_calvin[8] # torch.Size([4])
        his_vision_static = batch_calvin[9] # 4 12 20 1024
        his_pose = batch_calvin[10] # 4 12 20 3
        # print(idxes_sample)
        # merge the batch and the sequence dimension
        # images = images.flatten(0, 1)
        # gripper = gripper.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist: labels = labels[:, [-1]]  # only calculate last step action
        if args.fusion_mode == 'vit_concat': labels = labels[:, -1]
        labels = [labels[..., :6], labels[..., 6:]] # torch.Size([4, 13, 6]) torch.Size([4, 13, 1])

        if TESTTIME and torch.distributed.get_rank() == 0: ed_pd_yf = time.time(); print(f"YF: data prepress: {ed_pd_yf-st_yf} {idxes_sample}")
        assert args.use_static == False, "In navigation dataset, args.use_static must be False!!!"
        with autocast():
            output, static_pred, gripper_pred, obs_pred, aux_loss = model(
                vision_x=images[:, :args.window_size],
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper[:, :args.window_size],
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                calib = calib, # 注意里边实际上有args.window_size+1个
                pcd = pcd[:, :args.window_size], # UVFormer使用
                action_mask = action_mask,
                static_mask = static_mask,
                gripper_mask = gripper_mask,
                obs_mask = obs_mask,
                his_vision_static = his_vision_static,
                his_pose = his_pose,
                use_static = args.use_static, # action调用flamingo的时候，use_static为false效果最好
            )
     
        # np.save("tmp.npy", {"images": images[:,:12].cpu().numpy(), "input_ids": input_ids.cpu().numpy(), "attention_mask": attention_mask.cpu().numpy(), "gripper": gripper[:,:12].cpu().numpy(), "state_tensor": state_tensor.cpu().numpy() if (args.use_state or args.sep_lm_head) else None, "calib": calib, "action_mask": action_mask.cpu().numpy(), "static_mask": static_mask, "gripper_mask": gripper_mask, "obs_mask":obs_mask,"pcd":pcd.cpu().numpy(), "output": output, "static_pred": static_pred, "gripper_pred": gripper_pred})
        if TESTTIME and torch.distributed.get_rank() == 0: ed_fd_yf = time.time(); print(f"YF: model forward: {ed_fd_yf-ed_pd_yf}")
        loss_rgb, loss_gripper_rgb = torch.tensor(0), torch.tensor(0)
        if static_mask is not None :
            assert len(images[0]) == args.window_size + 1
            raw_rgb = rearrange(images[:, 1:], "B T F G D H W -> (B T F G) D H W")
            raw_rgb = F.interpolate(raw_rgb, (112, 112), mode='bilinear')
            rgbmask = torch.ones_like(raw_rgb)
            loss_rgb = MSE_Loss(static_pred, raw_rgb, rgbmask) * 0.1
        if gripper_mask is not None :
            raw_gripper = rearrange(gripper[:,1:], "B T F G D H W -> (B T F G) D H W")
            raw_gripper = F.interpolate(raw_gripper, (112, 112), mode='bilinear')
            grippermask = torch.ones_like(raw_gripper)
            loss_gripper_rgb = MSE_Loss(gripper_pred, raw_gripper, grippermask) * 0.1
            #loss_gripper_rgb = loss_gripper_rgb.sum() / (loss_gripper_rgb.shape[0] * loss_gripper_rgb.shape[1] * loss_gripper_rgb.shape[2] * loss_gripper_rgb.shape[3])

        loss_obs = defaultdict(float)
        if obs_mask is not None :
            pcd = rearrange(pcd[:, :args.window_size], "B T H W Z C  -> (B T) H W Z C")
            c_classes = pcd.shape[-1]
            grid_cls = ['occ','r','g','b']
            for ind in range(c_classes):
                preds_ind = obs_pred[:,:,:,:, ind]
                trues_ind = pcd[:,:,:,:, ind] 
                if ind == 0: 
                    loss_ind = balanced_bce_loss(preds_ind, trues_ind)
                else:
                    loss_ind = MSE_Loss(preds_ind, trues_ind, pcd[:,:,:,:, 0])
                loss_obs[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * args.occ_loss_weight[ind]
        if args.occ_loss and 'loss_occ' in aux_loss:
            loss_occ = aux_loss['loss_occ']
            for ind, k in enumerate(loss_occ.keys()):
                loss_obs[f"grid_cls_{k}_loss"] += loss_occ[k] * args.occ_loss_weight[ind]

        # compute loss
        if args.multi_action_token:
            num_actions, bin_actions = output.logits[0], output.logits[1]
            # reshape for loss calculation
            if args.multi_step_action != 1:
                assert False
                bs, seq_len = num_actions.shape[:2]
                num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            if 'Temporal' in args.fusion_mode:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][:, :args.window_size, :])
                num_classes = bin_actions.shape[-1] # bs t 20
                # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, torch.nn.functional.one_hot(labels[1][:, :args.window_size, 0].to(torch.int64), num_classes=num_classes).to(torch.float32)) # label onehot以后也是bs t 20
                # 调整预测张量的形状为 [bs * t, 20]，调整真实标签张量的形状为 [bs * t]
                loss_calvin_bin = torch.nn.functional.cross_entropy(bin_actions.view(-1, num_classes), labels[1][:, :args.window_size, 0].view(-1).to(torch.int64))
            else:
                assert False
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        elif args.train_action:
            assert False
            num_actions, bin_actions = output.logits[0], output.logits[1]
            # reshape for loss calculation
            if args.multi_step_action != 1:
                bs, seq_len = num_actions.shape[:2]
                num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            if 'Temporal' in args.fusion_mode:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][:,-1,:])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1][:,-1,:])
            else:
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        else:
            loss_calvin_num = torch.tensor(0).to(device_id)
            loss_calvin_bin = torch.tensor(0).to(device_id)
        # if args.real_data:
        #     loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        # else:
        #     loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
        loss_calvin = loss_calvin_num + loss_calvin_bin

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps
        #### BACKWARD PASS ####
        loss = (divided_loss_calvin * args.loss_multiplier_calvin)
        loss = loss * args.loss_weight['action']
        for k in loss_obs.keys():
            loss += loss_obs[k]*args.loss_weight['occ'] # YF: wight在这里乘的的，同时没有使用oRGB的wight
        loss = loss + loss_rgb + loss_gripper_rgb + loss_vqa

        mv_avg_loss.append(loss.item()) # 总loss，即机械臂loss+点云loss，后面没有使用
        calvin_avg_loss.append(loss_calvin.item()) # 机械臂loss
        if 'grid_cls_occ_loss' in loss_obs:
            occ_avg_loss.append(loss_obs['grid_cls_occ_loss'].item()) # 点云loss中的occ loss

        loss.backward()
        if TESTTIME and torch.distributed.get_rank() == 0: ed_bk_yf = time.time(); print(f"YF: backword: {ed_bk_yf - ed_fd_yf}")
        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
                m.weight.grad = m.weight.grad * zero_mask
        # model.apply(mask_embedding)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 模型中所有参数梯度的L2范数的总和会被裁剪到不超过1.0
        writer.add_scalar("totoal_norm",total_norm, num_steps)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (num_steps == num_batches_per_epoch - 1):

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], num_steps)
            writer.add_scalar("loss", loss.item(), num_steps)
            writer.add_scalar("loss_action", divided_loss_calvin.item(), num_steps)
            for k in loss_obs.keys():
                writer.add_scalar(k,loss_obs[k].item(), num_steps)
        if TESTTIME and torch.distributed.get_rank() == 0: ed_up_yf = time.time(); print(f"YF: update: {ed_up_yf - ed_bk_yf}")
        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}")
        # 计算最近100个或更少的损失值的平均值
        loss_dic = {"avg_action_loss": sum(calvin_avg_loss[-min(100,len(calvin_avg_loss)):]) / min(100,len(calvin_avg_loss)), "action_loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item(), "avg_occ_loss": sum(occ_avg_loss[-min(100,len(occ_avg_loss)):]) / (min(100,len(occ_avg_loss))+0.00001),}
        for k in loss_obs.keys():
            loss_dic[k] = loss_obs[k].item()
        loss_dic['lr'] = optimizer.param_groups[0]["lr"]
        loss_dic['succ'] = f"{(bin_actions.argmax(-1) == labels[1][:, :args.window_size, 0]).to(torch.float32).mean():.2f}"
        # loss_rgb + loss_gripper_rgb
        loss_dic['loss_rgb'] = loss_rgb.item()
        loss_dic['loss_gripper_rgb'] = loss_gripper_rgb.item()
        loss_dic['navvqa_avg_loss'] = sum(mv_avg_loss_vqa[-min(100,len(mv_avg_loss_vqa)):]) / min(100,len(mv_avg_loss_vqa))
        t.set_postfix(loss_dic)

        if not args.debug and args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
            if args.rank == 0:
                import os
                if not os.path.exists(args.run_name): os.makedirs(args.run_name)

                if args.delete_previous_checkpoint: 
                    import glob
                    ckpt_name = get_ckpt_name_pattern(args)
                    checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
                    # checkpoint_list = [_ for _ in checkpoint_list if "__sep" not in _ and 'iter' not in _ and 'weights' not in _]
                    if len(checkpoint_list) > 0:
                        print('args.delete_previous_checkpoint:', args.delete_previous_checkpoint)
                        for ckpt_path in sorted(checkpoint_list, key=os.path.getmtime)[:-3]: os.remove(ckpt_path)
                
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "cur_iter": num_steps + start_from_iter, # 保存ckpt的时候，需要cur_iter从current epoch的iter继续计算
                }
                ckpt_name = get_ckpt_name(args, global_step)
                ckpt_path = os.path.join(args.run_name, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint_dict, ckpt_path)
        # if global_step % 1000 == 0 and args.rank == 0 and global_step > 0:
        #     import os
        #     if not os.path.exists(args.run_name): os.makedirs(args.run_name)
        #     checkpoint_dict = {
        #         "epoch": epoch,
        #         "model_state_dict": get_checkpoint(model),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #     }
        #     ckpt_name = get_ckpt_name(args, 1001)
        #     ckpt_path = os.path.join(args.run_name, ckpt_name)
        #     print(f"Saving checkpoint to {ckpt_path}")
        #     torch.save(checkpoint_dict, ckpt_path)
        
        # gc.collect()
        # torch.cuda.empty_cache()
        if args.eval and evaluate_func is not None and args.eval_steps>0 and global_step % args.eval_steps == 0 and global_step > 0:
            evaluate_func(model)
        if TESTTIME and torch.distributed.get_rank() == 0: ed_yf = time.time(); print(f"YF: one iter: {ed_yf - st_yf}")
        # if global_step > 10: return


def val_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    writer
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    calvin_avg_loss = []
    occ_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        calib = batch_calvin[6]

        pcd = batch_calvin[7]
        
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        if 'Temporal' not in args.fusion_mode:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1) # input toekn重复12次
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        if 'Temporal' not in args.fusion_mode:
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1) # attention_mask重复12次
        else:
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True)
        
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        robot_obs = batch_calvin[5].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)

        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        if 'Temporal' not in args.fusion_mode:
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        if args.fusion_mode == 'vit_concat':
            labels = labels[:, -1]
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]

        with autocast():
            output,loss_occ = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                calib = calib,
                pcd = pcd,
            )



        # compute loss
        num_actions, bin_actions = output.logits[0], output.logits[1]

        # reshape for loss calculation
        if args.multi_step_action != 1:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)

        loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        if args.real_data:
            loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        else:
            loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps


        if 0:
            with torch.no_grad():
                import open3d as o3d
                import numpy as np
                from robouniview.data.preprocess_occ import OccupancyVFE
                voxel_range = [[-0.5, 0.5], [-0.5, 0.5], [0.3, 0.8]]
                voxel_size = [0.0125, 0.0125, 0.0125]
                vfe_generator = OccupancyVFE(voxel_range, voxel_size)
                occ = model.module.occ #(bs, w, h, z, c)
                occ_true = model.module.occ_true 
                occ[0][:,:,:,0] = torch.sigmoid(occ[0][:,:,:,0])
                occ = np.array(occ[0].cpu().detach())
                occ_true= np.array(occ_true[0].cpu())
                point,rgb = vfe_generator.decode_occupied_grid(occ)
                point_true,rgb_true = vfe_generator.decode_occupied_grid(occ_true)
                pcd_true = o3d.geometry.PointCloud()
                pcd_true.points = o3d.utility.Vector3dVector(point_true[:, :3])
                pcd_true.colors = o3d.utility.Vector3dVector(rgb_true)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                o3d.io.write_point_cloud("~/liufanfan/workspace/RoboFlamingo/pcd.pcd", pcd)
                o3d.io.write_point_cloud("~/liufanfan/workspace/RoboFlamingo/pcd_true.pcd", pcd_true)

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        for k in loss_occ.keys():
            loss += loss_occ[k]
       

        calvin_avg_loss.append(loss_calvin.item())
        if 'grid_cls_occ_loss' in loss_occ:
            occ_avg_loss.append(loss_occ['grid_cls_occ_loss'].item())
        #loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        # model.apply(mask_embedding)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        writer.add_scalar("totoal_norm",total_norm,num_steps)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # if args.rank == 0 and args.report_to_wandb:
            #     # compute within rank 0
            #     calvin_samples_per_second = (
            #         args.gradient_accumulation_steps
            #         * args.batch_size_calvin
            #         * args.world_size
            #         / step_time_m.val
            #     )
            #     calvin_samples_per_second_per_gpu = (
            #         args.gradient_accumulation_steps
            #         * args.batch_size_calvin
            #         / step_time_m.val
            #     )

                # api.log_metric_wandb("data_time",data_time_m.avg)
                # api.log_metric_wandb("step_time",step_time_m.avg)
                # api.log_metric_wandb("calvin_samples_per_second_per_gpu",calvin_samples_per_second_per_gpu)
                # api.log_metric_wandb("calvin_samples_per_second",calvin_samples_per_second)
                # api.log_metric_wandb("lr", optimizer.param_groups[0]["lr"])
            writer.add_scalar("lr",optimizer.param_groups[0]["lr"], num_steps)
            writer.add_scalar("loss",loss.item(),num_steps)
            writer.add_scalar("loss_action",divided_loss_calvin.item(),num_steps)
            for k in loss_occ.keys():
                writer.add_scalar(k,loss_occ[k].item(),num_steps)

                # step_time_m.reset()
                # data_time_m.reset()

                # api.log_metric_wandb("loss_calvin",divided_loss_calvin.item())
                # api.log_metric_wandb("global_step",global_step)


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
       
        loss_dic = {"val_epoch":epoch+1,"avg_action_loss": sum(calvin_avg_loss) / len(calvin_avg_loss), "action_loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item(), "avg_occ_loss": sum(occ_avg_loss) / (len(occ_avg_loss)+0.000001),}
        for k in loss_occ.keys():
            loss_dic[k] = loss_occ[k].item()

        loss_dic['lr'] = optimizer.param_groups[0]["lr"]
        t.set_postfix(loss_dic)


        #t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})
        
        # if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
                
        #     if args.rank == 0:
        #         import os
        #         if not os.path.exists(args.run_name):
        #             os.makedirs(args.run_name)

        #         checkpoint_dict = {
        #             "epoch": epoch,
        #             "model_state_dict": get_checkpoint(model),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #         }

        #         ckpt_name = get_ckpt_name(args, global_step)
        #         ckpt_path = os.path.join(args.run_name, ckpt_name)
        #         print(f"Saving checkpoint to {ckpt_path}")
        #         torch.save(checkpoint_dict, ckpt_path)
        #         if args.delete_previous_checkpoint:
        #             if epoch > 0:
        #                 os.remove(ckpt_path)

def train_one_epoch_calvin_cotrain(
    args,
    model,
    epoch,
    calvin_loader,
    coco_loader,
    vqa_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # setup loaders
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    t = tqdm(
        enumerate(zip(coco_loader, vqa_loader, calvin_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")

    mv_avg_loss = []
    mv_avg_loss_coco = []
    mv_avg_loss_vqa = []
    for num_steps, (batch_coco, batch_vqa, batch_calvin) in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### COCO FORWARD PASS ####
        images = batch_coco[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch_coco[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_coco[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        def calculate_vl_cross_entropy(logits, labels, mask=None):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if mask is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(
                        -1, logits.shape[-1]
                    ),
                    shift_labels.view(-1),
                )
            else:
                # TODO: mask is with the same shape of labels, 
                # 1 represents valid, 0 for non-valid, only calculate loss for valid tokens
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                shift_logits.view(
                        -1, logits.shape[-1]
                    ),
                shift_labels.view(-1),
                )
                # mask the loss
                mask = mask[..., 1:].contiguous()
                loss = loss * mask.reshape(-1)
                # mean
                loss = loss.mean()
            return loss

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                mode = 'vision_lang'
            )
        
        logits = output.logits
        loss_coco = calculate_vl_cross_entropy(logits, labels)
        mv_avg_loss_coco.append(loss_coco.item())
        divided_loss_coco = loss_coco * args.vl_task_weights
        divided_loss_coco = divided_loss_coco / args.gradient_accumulation_steps
        
        (divided_loss_coco * args.loss_multiplier_calvin).backward()

        #### VQA FORWARD PASS ####
        images = batch_vqa[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch_vqa[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_vqa[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )
        ques_mask = batch_vqa[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                # labels=labels,
                mode = 'vision_lang'
            )
        
        logits = output.logits
        loss_vqa = calculate_vl_cross_entropy(logits, labels, ques_mask)
        mv_avg_loss_vqa.append(loss_vqa.item())
        divided_loss_vqa = loss_vqa * 0.5
        divided_loss_vqa = divided_loss_vqa / args.gradient_accumulation_steps
        (divided_loss_vqa * args.loss_multiplier_calvin).backward()
        
        #### CALVIN FORWARD PASS ####
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        if 'Temporal' not in args.fusion_mode:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        if 'Temporal' not in args.fusion_mode:
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True)
        
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        robot_obs = batch_calvin[5].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)

        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        if 'Temporal' not in args.fusion_mode:
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        if 'Temporal' not in args.fusion_mode:
            labels = labels[:, -1]
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]

        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            )

        # compute loss
        num_actions, bin_actions = output.logits[0], output.logits[1]

        def discretize_actions(pose_action):
            action_min = -1.001
            action_max = 1.001
            action_len = (action_max - action_min) / args.act_disc
            pose_action = (pose_action - action_min) / action_len
            pose_action = torch.floor(pose_action).long()
            return pose_action
        
        if args.act_disc != -1:
            # assert labels[0].max() < 1.0, f"{labels[0].max()} >= 1.0"
            # assert labels[0].min() > -1.0, f"{labels[0].min()} <= -1.0"
            labels[0] = discretize_actions(labels[0])
            assert labels[0].max() < args.act_disc, f"{labels[0].max()} >= {args.act_disc}"
            assert labels[0].min() >= 0, f"{labels[0].min()} < 0"
        # reshape for loss calculation
        if args.multi_step_action != 1:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)

        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        if args.act_disc == -1:
            loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
            if args.real_data:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
            else:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
        else:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.view(bs, seq_len, -1, args.act_disc).permute(0, 3, 1, 2)
            labels[0] = labels[0].view(bs, seq_len, -1)
            # print('-'*100)
            # print(num_actions, labels[0])
            loss_calvin_num = torch.nn.functional.cross_entropy(num_actions, labels[0])
            if args.real_data:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.2
            else:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.1
        

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()
        
        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                coco_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    * args.world_size
                    / step_time_m.val
                )
                coco_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    / step_time_m.val
                )
                vqa_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    * args.world_size
                    / step_time_m.val
                )
                vqa_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    / step_time_m.val
                )
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "coco_samples_per_second": coco_samples_per_second,
                        "coco_samples_per_second_per_gpu": coco_samples_per_second_per_gpu,
                        "vqa_samples_per_second": vqa_samples_per_second,
                        "vqa_samples_per_second_per_gpu": vqa_samples_per_second_per_gpu,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_coco": loss_coco.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_vqa": loss_vqa.item(), "global_step": global_step},
                    commit=False,
                )

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete.  Loss coco: {loss_coco.item():.3f} // Loss vqa: {loss_vqa.item():.3f} // Loss CALVIN: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg calvin loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "avg coco loss": sum(mv_avg_loss_coco[-avg_horizon:]) / avg_horizon, "avg vqa loss": sum(mv_avg_loss_vqa[-avg_horizon:]) / avg_horizon,
                        "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})


def train_one_epoch_calvin_two_way(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        # images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(1).unsqueeze(1))
        vision_x = torch.cat([images, gripper], dim=0)
        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(2, images.shape[1], 1)

        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(2, images.shape[1], 1)
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True).repeat(2, 1, 1).unsqueeze(2).unsqueeze(2)
        # import pdb; pdb.set_trace()
        # merge the batch and the sequence dimension
        # images = images.flatten(0, 1)
        # gripper = gripper.flatten(0, 1)
        images = images.detach().cpu()
        gripper = gripper.detach().cpu()
        vision_x = vision_x.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        input_ids = input_ids.flatten(0, 1)
        attention_mask = attention_mask.flatten(0, 1)

        # attention_mask = batch_calvin[1][1].to(device_id, dtype=cast_dtype, non_blocking=True)
        # attention_mask = None

        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]
        # labels = [labels[..., :6], labels[..., 6:]]

        with autocast():
            output = model(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=None,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            )

        # compute loss
        num_actions, bin_actions = output.logits
        loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        # loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})


def train_one_epoch(
    args,
    model,
    epoch,
    laion_loader,
    mmc4_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_mmc4 = mmc4_loader.num_batches

    assert (
        num_batches_per_epoch_laion == num_batches_per_epoch_mmc4
    ), "Number of batches in laion and mmc4 datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_mmc4
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_laion, batch_mmc4) in tqdm(
        enumerate(zip(laion_loader, mmc4_loader)),
        # disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch

        #### LAION FORWARD PASS ####
        images = (
            batch_laion[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        input_ids = batch_laion[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_laion[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        divided_loss_laion = loss_laion / args.gradient_accumulation_steps

        #### C4 FORWARD PASS ####
        images = (
            batch_mmc4[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(2)
        )
        input_ids = torch.stack([x[0] for x in batch_mmc4[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_mmc4[1]]).squeeze(1)

        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100

        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_mmc4 = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

            # if loss is nan, skip this batch
            if torch.isnan(loss_mmc4):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad()
                continue

        divided_loss_mmc4 = loss_mmc4 / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_laion * args.loss_multiplier_laion
            + divided_loss_mmc4 * args.loss_multiplier_mmc4
        )
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )

                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_laion": divided_loss_laion.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_mmc4": divided_loss_mmc4.item(), "global_step": global_step},
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0):
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss LAION: {loss_laion.item():.3f} // Loss MMC4: {loss_mmc4.item():.3f}"
            )


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    return state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
