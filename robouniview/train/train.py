""" Main training script """
import sys
import os  
env = os.environ    
current_path = os.getcwd()  
robouniview_path =  current_path  
env['PATH'] = env['PATH'] + ':'+  robouniview_path
sys.path.append(robouniview_path)
sys.path.append('~/liufanfan/workspace/RoboUniView/open_flamingo')
import argparse
import copy
import glob
import os
sys.path.append('~/liufanfan/workspace/RoboUniView/open_flamingo')
# sys.path.append('~/liufanfan/workspace/calvin/calvin_models')
# sys.path.append('~/liufanfan/workspace/calvin/calvin_env')
# sys.path.append('~/liufanfan/workspace/calvin/calvin_env/tacto_env')
sys.path.append('~/yanfeng/project/robotic/calvin/calvin_models')
sys.path.append('~/yanfeng/project/robotic/calvin/calvin_env')
sys.path.append('~/yanfeng/project/robotic/calvin/calvin_env/tacto_env')
os.environ['TORCH_HOME']='~/yanfeng/project/robotic/modelzoo/'
os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '~/yanfeng/project/robotic/Metaworld/mujoco210'
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
from huggingface_hub import hf_hub_download
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from robouniview.data.multi_cam_data import get_data
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from train_utils import get_checkpoint, train_one_epoch_calvin, train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
get_ckpt_name, get_ckpt_name_pattern, train_one_epoch_move, train_one_epoch_nav_vqa
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from robouniview.models.factory import create_model_and_transforms, mpt_dict
import yaml
from robouniview.eval import eval_one_epoch_ddp
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math
import time

os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
print(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr:float = 0.1, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps)) # 这里的结果会和base_lr相乘，因此必须以1为基准来浮动base_lr
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)*(1-min_lr) +  min_lr)) # 这里的结果会和base_lr相乘，因此必须以1为基准来浮动base_lr
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class GlobalConfig:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self._load_config()

    def _load_config(self):
        for key, val in self.config.items():
            setattr(self, key, val)


def load_global_config_yaml_only(config_path: str) -> GlobalConfig:
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)
    global_config = GlobalConfig(config)
    return global_config

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    print('excutin')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="save_dir",
    )
    args_ = parser.parse_args()
    args = load_global_config_yaml_only(args_.config)

    attributes_and_values = vars(args)
    attributes_and_values = dict(attributes_and_values)
    if args_.save_dir is not None: args.save_dir = args_.save_dir
    args.run_name = args.save_dir
    if not os.path.exists(args.save_dir):
        try: os.makedirs(args.save_dir)
        except: pass
    with open(args.save_dir+'/config.yaml', 'w') as f:
        yaml.dump(attributes_and_values['config'], f, default_flow_style=None, sort_keys=False)

    # window_size:
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion": args.eval_hist_size = args.n_obs_steps
    if args.tcp_rel: args.clip_state = True
    if args.save_checkpoints_to_wandb and not args.report_to_wandb: raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # args.rank = 0
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    random_seed(args.seed)
    args.lm_path = mpt_dict[args.llm_name]["lang_encoder_path"]
    args.tokenizer_path = mpt_dict[args.llm_name]["tokenizer_path"]
    args.cross_attn_every_n_layers = mpt_dict[args.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.llm_name]["openflamingo_checkpoint"]

    time.sleep(args.local_rank)
    print(f"sleep: {args.local_rank} s")

    model, image_processor, tokenizer = create_model_and_transforms(
        args,
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        fusion_mode=args.fusion_mode,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"), # Diff still have bugs of loaded data mismatch
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        predict_epsilon=args.predict_epsilon,
        sep_lm_head=args.sep_lm_head,
        unfreeze_vit=args.unfreeze_vit,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        pooling=args.pooling,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
        decoder_type=args.decoder_type,
        hidden_size=args.hidden_size,
        freeze_sampler=args.freeze_sampler,
        fwd_pred=args.fwd_pred,
        fwd_pred_hand=args.fwd_pred_hand,
        no_image_patch=args.no_image_patch,
        global_latent=args.global_latent,
        clip_cache_dir=args.clip_cache_dir,
        nclass_gripper=args.nclass_gripper if hasattr(args, 'nclass_gripper') else 1,
    )

    checkpoint_path = args.openflamingo_checkpoint
    if not args.debug and not args.no_pretrain:
        checkpoint = torch.load(checkpoint_path)
        if model.lang_encoder.transformer.wte.weight.shape == checkpoint['lang_encoder.transformer.wte.weight'].shape:
            model.load_state_dict(checkpoint, strict=False)
        else:
            random_wte = model.lang_encoder.transformer.wte.weight.data
            random_wte[:checkpoint['lang_encoder.transformer.wte.weight'].shape[0],:] = checkpoint['lang_encoder.transformer.wte.weight']
            checkpoint['lang_encoder.transformer.wte.weight'] = random_wte
            model.load_state_dict(checkpoint, strict=False)
        if args.residual: model.lang_encoder.clone_parameters()
    print(f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    calvin_dataset = get_data(args, image_processor, tokenizer, "all")

    if hasattr(args, "use_vqa") and args.use_vqa:
        vqa_dataset = get_data(args, image_processor, tokenizer, ["spocvqa"])

    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")

    #if args.rank == 0 and args.report_to_wandb:
        # wandb.init(
        #     project=args.wandb_project,
        #     # entity=args.wandb_entity,
        #     name=args.run_name,
        #     config=vars(args),
        # )
    writer = SummaryWriter(os.path.join(args.save_dir ,'log'))

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    if args.head_type == "diffusion" and (not args.debug):
        normalizer = model.diffusion_model.normalizer
        all_actions = np.vstack([calvin_dataset.dataset.__getitem__((i,1),True)["actions"] for i in range(0,10000)])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')

    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    ddp_model._set_static_graph()  # 告知DDP模型图是静态的/实验性功能

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []
        def apply_decay(x):
            return (
                ("gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x)
                or ("uvformer"in x)
                or ("Upsample2d_3d"in x)
                or ("alignment_layer" in x)
                or ("occ_decoder" in x)
            )
        for n, p in model.named_parameters():
            if apply_decay(n):
                params_with_wd.append(p)
                # print(n)
            else:
                params_without_wd.append(p)
        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": args.weight_decay},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]
    args.learning_rate = args.learning_rate # * (args.world_size / 8)
    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)
    # total_training_steps = (
    #     (args.train_num_samples_calvin) // (args.batch_size_calvin * args.world_size)
    # ) * args.num_epochs
    # 计算iter数
    num_batches_per_epoch_calvin = calvin_dataset.dataloader.num_batches # num_batches是考虑过world_size的
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
            min_lr = 0.01,
        )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    use_diff = (args.head_type == "diffusion")
    # check if a checkpoint exists for this run
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        ckpt_name = get_ckpt_name_pattern(args)
        checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
        print("ckpt_name:", ckpt_name)

        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            checkpoint_list = sorted(checkpoint_list, key=os.path.getmtime, reverse=True)
            expected_size = 18000000000 if args.precision != "bf16" else 9000000000

            # 对 checkpoint_list 进行循环尝试加载
            for checkpoint_path in checkpoint_list:
                try:
                    actual_size = os.path.getsize(checkpoint_path)
                    if actual_size > expected_size:
                        args.resume_from_checkpoint = checkpoint_path
                        print(f"Successfully found valid checkpoint {args.resume_from_checkpoint} for run {args.run_name}.")
                        print(f"load from iter {args.resume_from_checkpoint.split('_')[-2].split('.')[0]} on all epochs")
                        break  # 成功找到有效的 checkpoint 后退出循环
                    else:
                        print(f"Checkpoint {checkpoint_path} size mismatch: expected {expected_size}, got {actual_size}.")
                except OSError as e:
                    print(f"Error accessing file {checkpoint_path}: {e}")
            
            # 如果没有任何 checkpoint 加载成功
            if args.resume_from_checkpoint is None:
                print(f"Found no valid checkpoints for run {args.run_name}.")

    resume_from_epoch = 0
    start_from_iter = 0
    if args.load_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.load_from_checkpoint}")
        checkpoint = torch.load(args.load_from_checkpoint, map_location="cpu")
        
        def filter_ckpt(checkpoint, flags=[]):
            new_state_dict = OrderedDict()
            for key, value in checkpoint["model_state_dict"].items():
                load_p = True
                for flag in flags:
                    if flag in key: load_p = True
                if load_p:
                    if 'bevformer' in key: key = key.replace('bevformer','uvformer')
                    if 'bev2vision' in key: key = key.replace('bev2vision','alignment_layer')
                    new_state_dict[key] = value
            return new_state_dict
        flags =['bevformer','bevformer_gripper','Upsample2d_3d','linear']
        # ddp_model.load_state_dict(filter_ckpt(checkpoint, flags), False)
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)

    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        print(checkpoint.keys(), 'args.from_scratch=', args.from_scratch, 'args.real_data=', args.real_data, "args.continue_iter", args.continue_iter)

        if args.continue_iter and "cur_iter" in checkpoint.keys(): # 可以通过continue_iter参数，手动控制
            # 如果使用continue_iter，那么在本epoch下继续，resume_from_epoch不+1
            start_from_iter = checkpoint.get("cur_iter", 0)
            resume_from_epoch = checkpoint.get("epoch", 0)
        else:
            # 如果不使用continue_iter，那么在下一个epoch下继续，resume_from_epoch+1
            resume_from_epoch = checkpoint.get("epoch", 0) + 1

        if not args.real_data:
            try:
                if not args.reset_lr: # reset_lr为true，不使用ckpt的lr，使用args.learning_rate
                    if "optimizer_state_dict" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if "lr_scheduler_state_dict" in checkpoint:
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

                if args.reset_epoch != -1: # reset_epoch为非负数时，不使用ckpt的epoch，使用指定的epoch开始
                    resume_from_epoch = args.reset_epoch

            except KeyError as e:
                print(f"Missing key in checkpoint: {e}")

        print('start_from_iter', start_from_iter, 'resume_from_epoch', resume_from_epoch)

    ddp_model.train()
    if args.real_data: resume_from_epoch = 0
    def evaluate_func(ddp_model):
        eval_log_dir = args.save_dir
        ddp_model.eval()
        eval_one_epoch_ddp(
            args=args,
            model=ddp_model,
            image_processor=image_processor,
            tokenizer=tokenizer,
            dataset_path=args.calvin_dataset,
            future_act_len=args.future_act_len,
            eval_log_dir=eval_log_dir,
            reset=args.reset,
            diverse_inst=args.diverse_inst,
            debug=True,
        )
        ddp_model.train()
    evaluate_f = lambda ddp_model: evaluate_func(ddp_model)
    if hasattr(args, 'eval_first') and args.eval_first: evaluate_f(ddp_model)
    for epoch in range(resume_from_epoch, args.num_epochs):
        ddp_model.train()
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader
        if hasattr(args, "use_vqa") and args.use_vqa:
            global_step = epoch * num_batches_per_epoch_calvin + start_from_iter # 从all epochs的start from iter计算总iter

            num_batches_per_epoch_vqa = vqa_dataset.dataloader.num_batches
            vqa_epoch = global_step // num_batches_per_epoch_vqa
            vqa_start_from_iter = global_step % num_batches_per_epoch_vqa

            vqa_dataset.set_epoch(vqa_epoch)
            vqa_loader = vqa_dataset.dataloader

            train_one_epoch_nav_vqa(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                vqa_loader=vqa_loader,
                device_id=device_id,
                wandb=wandb,
                writer=writer,
				evaluate_func=evaluate_f,
                start_from_iter=start_from_iter,
                vqa_start_from_iter=vqa_start_from_iter,
                vqa_dataset=vqa_dataset,
            )
        elif args.head_type == "diffusion":
            train_one_epoch_calvin_diff(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif args.fusion_mode == 'two_way':
            train_one_epoch_calvin_two_way(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif 'move' in args.fusion_mode:
            train_one_epoch_move(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
                writer=writer,
				evaluate_func=evaluate_f,
                start_from_iter=start_from_iter,
            )
        else:
            train_one_epoch_calvin(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
                writer=writer,
				evaluate_func=evaluate_f,
                start_from_iter=start_from_iter,
            )
        if args.rank == 0:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            if not args.debug:
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(ddp_model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }
                ckpt_name = get_ckpt_name(args, epoch)
                ckpt_path = os.path.join(args.save_dir, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint_dict, ckpt_path)
                if args.delete_previous_checkpoint:
                    print('args.delete_previous_checkpoint')
                    if epoch > 0: os.remove(ckpt_path)
        start_from_iter = 0 # 只有load ckpt的时候跳过iter，后续不跳过
        if args.eval: evaluate_f(ddp_model)

    if args.rank == 0:
        if not os.path.exists(args.save_dir): 
            os.makedirs(args.save_dir)
        if not args.debug:
            ckpt_name = get_ckpt_name(args,)
            torch.save(get_checkpoint(ddp_model), f"{args.save_dir}/{ckpt_name}")
        # if args.report_to_wandb and args.save_checkpoints_to_wandb:
            # wandb.save(f"{args.run_name}/{ckpt_name}")
    # 关闭wandb
    writer.close()


if __name__ == "__main__":

    main()

    # def rename_state_dict_keys(state_dict, old_prefix, new_prefix):
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         if key.startswith(old_prefix):
    #             new_key = new_prefix + key[len(old_prefix):]
    #         else:
    #             new_key = key
    #         new_state_dict[new_key] = value
    #     return new_state_dict
    # checkpoint['model_state_dict'] = rename_state_dict_keys(checkpoint['model_state_dict'], 'module.occ_decoder', 'module.occ_decoder_UVFormer')
    # checkpoint['model_state_dict'] = rename_state_dict_keys(checkpoint['model_state_dict'], 'module.Upsample2d_3d', 'module.Upsample2d_3d_UVFormer')
