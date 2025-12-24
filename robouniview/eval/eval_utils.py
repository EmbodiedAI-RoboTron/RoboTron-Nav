import argparse
from collections import Counter, defaultdict, namedtuple
import logging
import os, json, random, pickle
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
import torch.distributed as dist
from moviepy.editor import ImageSequenceClip
from robouniview.data.preprocess_occ import OccupancyVFE
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from einops import rearrange
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
import open3d as o3d
from calvin_env.envs.play_table_env import get_env
from robouniview.data.multi_cam_data import preprocess_image, preprocess_text_calvin
from robouniview.utils import world_to_tcp_frame, tcp_to_world_frame
import functools
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000
# NUM_SEQUENCES = 400
import pybullet as pb
import cv2

def get_gripper_camera_view_matrix(cam):
    camera_ls = pb.getLinkState(
        bodyUniqueId=cam.robot_uid,
        linkIndex=cam.gripper_cam_link,
        physicsClientId=cam.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = pb.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = pb.computeViewMatrix(
        camera_pos, camera_pos + cam_rot_y, -cam_rot_z
    )
    return view_matrix

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

class DebugEnv():
    
    def __init__(self) -> None:
        pass
    
    def get_random_obs(self):
        obs = {}
        obs['rgb_obs'] = {}
        obs['rgb_obs']['rgb_static'] = np.ones((200, 200, 3), dtype=np.uint8)
        obs['rgb_obs']['rgb_gripper'] = np.ones((84, 84, 3), dtype=np.uint8)
        obs['robot_obs'] = np.ones(15, dtype=np.float32)
        return obs
    
    def get_obs(self):
        return self.get_random_obs()
    
    def step(self, action):
        return self.get_random_obs()

    def reset(self, **kwargs):
        return

    def get_info(self):
        return


def make_env_debug(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class ModelWrapper(CalvinBaseModel):
    def __init__(self,args, model, tokenizer, image_processor, cast_dtype, use_diff, history_len=None, future_act_len=-1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.replan = model.module.replan
        self.decoder_type = model.module.decoder_type
        self.cast_type = cast_dtype
        self.use_diff = use_diff
        # self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer, action_token = args.action_token,  multi_action_token=args.multi_action_token)
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer, sample_mode=args.sample_mode) # 注意此处是输出图片+OCC+action
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.feature_cache = None
        self.dt_feat_cache = []
        self.fusion_mode = self.model.module.fusion_mode
        self.args = args
        
        if use_diff:
            self.diffusion_model = None
            self.normalizer = None
            if isinstance(self.model, DistributedDataParallel):
                self.diffusion_model = self.model.module.diffusion_model
            else:
                self.diffusion_model = self.model.diffusion_model
            action_dim = self.diffusion_model.data_dim
            horizon = self.diffusion_model.horizon
            self.normalizer = self.diffusion_model.normalizer
            self.action_hist_queue = deque(maxlen=history_len-1)
            self.action_hist_queue.extend([np.zeros(action_dim) for _ in range(history_len-1)])

            if horizon-history_len+1:
                self.supp = None
            self.hist_len = history_len-1
            self.action_dim = action_dim
            self.horizon = horizon
            self.future_act_len = future_act_len
        
        # if self.model.module.pad_length != -1:
        if self.model.module.pad_length == -1:
            history_len = self.model.module.window_size
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.calib_queue = deque(maxlen=history_len)

    def reset(self):
        """
        This is called
        """
        if self.use_diff:
            self.action_hist_queue = deque(maxlen=self.hist_len)
            self.action_hist_queue.extend([np.zeros(self.action_dim) for _ in range(self.hist_len)])
        if self.model.module.pad_length != -1:
            history_len = self.model.module.pad_length
        else:
            history_len = self.model.module.window_size
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.calib_queue = deque(maxlen=history_len)
        self.feature_cache = None
        self.dt_feat_cache = []
        
        self.model.module.lang_encoder.lm_head.hidden_state = None
        self.model.module.lang_encoder.lm_head.history_memory = []

        if self.model.module.sep_lm_head:
            self.model.module.lm_head.hidden_state = None
            self.model.module.lm_head.history_memory = []

    def step(self, obs, goal,env, get_action=True):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """

        static_cam_env = env.cameras[0]
        gripper_cam_env = env.cameras[1]
        gripper_cam_env.viewMatrix = get_gripper_camera_view_matrix(gripper_cam_env)


        static_extrinsic = np.array(static_cam_env.viewMatrix).reshape((4, 4)).T
        gripper_extrinsic = np.array(gripper_cam_env.viewMatrix).reshape((4, 4)).T
        static_foc = static_cam_env.height / (2 * np.tan(np.deg2rad(static_cam_env.fov) / 2))
        gripper_foc = gripper_cam_env.height / (2 * np.tan(np.deg2rad(gripper_cam_env.fov) / 2))
        static_intrinsic = np.array([[static_foc , 0.0, static_cam_env.height/2], [0.0, static_foc , static_cam_env.height/2], [0.0, 0.0, 1.0]])
        gripper_intrinsic = np.array([[gripper_foc , 0.0, gripper_cam_env.height/2], [0.0, gripper_foc , gripper_cam_env.height/2], [0.0, 0.0, 1.0]])

        calib = {'rgb_static':{'extrinsic_matrix':static_extrinsic,
                                'intrinsic_matrix':static_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                'cam_config':{"height": static_cam_env.height, "width": static_cam_env.width, "fov": static_cam_env.fov}},
                'rgb_gripper':{'extrinsic_matrix':gripper_extrinsic,
                                'intrinsic_matrix':gripper_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                'cam_config':{"height": gripper_cam_env.height, "width": gripper_cam_env.width, "fov": gripper_cam_env.fov}}}
        static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
        gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']
        # 平移矩阵 T_translate # 原始是0.3-0.8因此无需添加平移矩阵
        if 0:
            from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, RandomShiftsAug
            static_cam = cam(static_extrinsic_matrix, calib['rgb_static']['cam_config']['height'], calib['rgb_static']['cam_config']['width'], calib['rgb_static']['cam_config']['fov'])
            gripper_cam = cam(gripper_extrinsic_matrix,  calib['rgb_gripper']['cam_config']['height'],  calib['rgb_gripper']['cam_config']['width'],  calib['rgb_gripper']['cam_config']['fov'])

            rgb_static, depth_static = obs["rgb_obs"]['rgb_static'], obs["depth_obs"]['depth_static']
            rgb_gripper, depth_gripper = obs["rgb_obs"]['rgb_gripper'], obs["depth_obs"]['depth_gripper']

            static_pcd = deproject(
                static_cam, depth_static,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            gripper_pcd = deproject(
                gripper_cam, depth_gripper,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            rgb_static = rgb_static.reshape(-1, 3)/255.
            rgb_gripper = rgb_gripper.reshape(-1, 3)/255.

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(rgb_static)
            o3d.io.write_point_cloud("tmp.pcd", pcd)
            pcd.points = o3d.utility.Vector3dVector(gripper_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(rgb_gripper)
            o3d.io.write_point_cloud("tmp1.pcd", pcd)

        calib['static_extrinsic_matrix'] = static_extrinsic_matrix*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        calib['static_intrinsic_matrix'] = calib['rgb_static']['intrinsic_matrix']
        calib['static_distCoeffs_matrix'] = calib['rgb_static']['distCoeffs_matrix']

        calib['gripper_extrinsic_matrix'] = gripper_extrinsic_matrix*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        calib['gripper_intrinsic_matrix'] = calib['rgb_gripper']['intrinsic_matrix']
        calib['gripper_distCoeffs_matrix'] = calib['rgb_gripper']['distCoeffs_matrix']

        static_extrinsic_matrix = torch.from_numpy(calib['static_extrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        static_intrinsic_matrix = torch.from_numpy(calib['static_intrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        static_distCoeffs_matrix = torch.from_numpy(calib['static_distCoeffs_matrix']).unsqueeze(0).unsqueeze(0)
        gripper_extrinsic_matrix = torch.from_numpy(calib['gripper_extrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        gripper_intrinsic_matrix = torch.from_numpy(calib['gripper_intrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        gripper_distCoeffs_matrix = torch.from_numpy(calib['gripper_distCoeffs_matrix']).unsqueeze(0).unsqueeze(0)
        static_fov = torch.from_numpy(np.array([calib['rgb_static']['cam_config']['fov'], calib['rgb_static']['cam_config']['height'], calib['rgb_static']['cam_config']['width']])).unsqueeze(0).unsqueeze(0)
        gripper_fov = torch.from_numpy(np.array([calib['rgb_gripper']['cam_config']['fov'], calib['rgb_gripper']['cam_config']['height'],  calib['rgb_gripper']['cam_config']['width']])).unsqueeze(0).unsqueeze(0)
        calib = (static_extrinsic_matrix,static_intrinsic_matrix,static_distCoeffs_matrix,
                 gripper_extrinsic_matrix,gripper_intrinsic_matrix,gripper_distCoeffs_matrix,gripper_extrinsic_matrix,static_fov,gripper_fov)

        # preprocess image
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        
        # fix window_size : ddp_model -> ... -> window_size
        if self.model.module.sep_lm_head:
            window_size = self.model.module.lm_head.window_size
            self.model.module.lm_head.window_size = 1
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                self.model.module.lm_head.window_size = self.model.module.pad_length
        else:
            window_size = self.model.module.lang_encoder.lm_head.window_size
            self.model.module.lang_encoder.lm_head.window_size = 1
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                self.model.module.lang_encoder.lm_head.window_size = self.model.module.pad_length
        gripper = None
        state = None

        if self.model.module.use_gripper:
            gripper = obs["rgb_obs"]['rgb_gripper']
            gripper = Image.fromarray(gripper)
            gripper = self.image_process_fn([gripper])
            # expand image dimension
            gripper = gripper.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        
        # if self.model.module.use_state or self.model.module.sep_lm_head:
        if self.model.module.use_state or self.model.module.sep_lm_head:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state]))
            # if self.model.module.sep_lm_head:
            #     state = torch.cat([state[...,:6], state[...,[-1]]], dim=-1)
            if self.fusion_mode == 'two_way':
                state = state.repeat(2, 1)
            state = state.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
            state = state.to(torch.float32)
        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            if gripper is not None:
                gripper = gripper.to(device)
            if state is not None:
                state = state.to(device)

            if 'Temporal' in self.fusion_mode:
                self.model.module.pad_length = self.model.module.window_size

            self.model.module.pad_length = -1   # YF:
            # if self.model.module.pad_length != -1:
            if len(self.img_queue) == 0:
                self.img_queue.append(image_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.img_queue.append(image_x)
            else:
                self.img_queue.append(image_x)
            if len(self.gripper_queue) == 0 and gripper is not None:
                self.gripper_queue.append(gripper)
                for _ in range(self.model.module.pad_length - 1):
                    self.gripper_queue.append(gripper)
            else:
                self.gripper_queue.append(gripper)
            if len(self.state_queue) == 0 and state is not None:
                self.state_queue.append(state)
                for _ in range(self.model.module.pad_length - 1):
                    self.state_queue.append(state)
            else:
                self.state_queue.append(state)
            if len(self.calib_queue) == 0:
                self.calib_queue.append(calib)
                for _ in range(self.model.module.pad_length - 1):
                    self.calib_queue.append(calib)
            else:
                self.calib_queue.append(calib)

            if 'Temporal' in self.fusion_mode:
                image_x = torch.cat(list(self.img_queue), dim=1)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=1)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=1)

                image_x = image_x.unsqueeze(2)
                gripper = gripper.unsqueeze(2)
                calib = [torch.cat([tmp[i_calibe] for tmp in self.calib_queue], dim=1) for i_calibe in range(len(self.calib_queue[0]))]

            
            self.model.module.pad_length = -1

            # expand text dimension
            text_x, mask = self.text_process_fn([goal], window_size=len(self.img_queue))
            text_x = text_x.to(device)
            mask = mask.to(device)
            
            action_token_id = self.tokenizer("<action>", add_special_tokens=False)["input_ids"][-1]
            action_mask = mask.clone()
            action_mask[text_x != action_token_id] = 0

            # preimg0_token_id = self.tokenizer("<preimg0>", add_special_tokens=False)["input_ids"][-1]
            # preimg1_token_id = self.tokenizer("<preimg1>", add_special_tokens=False)["input_ids"][-1]
            # preimg2_token_id = self.tokenizer("<preimg2>", add_special_tokens=False)["input_ids"][-1]
            # preimg3_token_id = self.tokenizer("<preimg3>", add_special_tokens=False)["input_ids"][-1]
            # preimg4_token_id = self.tokenizer("<preimg4>", add_special_tokens=False)["input_ids"][-1]
            # preimg5_token_id = self.tokenizer("<preimg5>", add_special_tokens=False)["input_ids"][-1]
            # preimg6_token_id = self.tokenizer("<preimg6>", add_special_tokens=False)["input_ids"][-1]
            # preimg7_token_id = self.tokenizer("<preimg7>", add_special_tokens=False)["input_ids"][-1]
            # action_token_id = tokenizer("<action>", add_special_tokens=False)["input_ids"][-1]
        
            static0 = self.tokenizer("<static0>", add_special_tokens=False)["input_ids"][-1]
            static1 = self.tokenizer("<static1>", add_special_tokens=False)["input_ids"][-1]
            static2 = self.tokenizer("<static2>", add_special_tokens=False)["input_ids"][-1]
            static3 = self.tokenizer("<static3>", add_special_tokens=False)["input_ids"][-1]
            static4 = self.tokenizer("<static4>", add_special_tokens=False)["input_ids"][-1]
            static5 = self.tokenizer("<static5>", add_special_tokens=False)["input_ids"][-1]
            static6 = self.tokenizer("<static6>", add_special_tokens=False)["input_ids"][-1]
            static7 = self.tokenizer("<static7>", add_special_tokens=False)["input_ids"][-1]
            
            gripper0 = self.tokenizer("<gripper0>", add_special_tokens=False)["input_ids"][-1]
            gripper1 = self.tokenizer("<gripper1>", add_special_tokens=False)["input_ids"][-1]
            gripper2 = self.tokenizer("<gripper2>", add_special_tokens=False)["input_ids"][-1]
            gripper3 = self.tokenizer("<gripper3>", add_special_tokens=False)["input_ids"][-1]
            gripper4 = self.tokenizer("<gripper4>", add_special_tokens=False)["input_ids"][-1]
            gripper5 = self.tokenizer("<gripper5>", add_special_tokens=False)["input_ids"][-1]
            gripper6 = self.tokenizer("<gripper6>", add_special_tokens=False)["input_ids"][-1]
            gripper7 = self.tokenizer("<gripper7>", add_special_tokens=False)["input_ids"][-1]
            
            obs0 = self.tokenizer("<obs0>", add_special_tokens=False)["input_ids"][-1]
            obs1 = self.tokenizer("<obs1>", add_special_tokens=False)["input_ids"][-1]
            obs2 = self.tokenizer("<obs2>", add_special_tokens=False)["input_ids"][-1]
            obs3 = self.tokenizer("<obs3>", add_special_tokens=False)["input_ids"][-1]
            obs4 = self.tokenizer("<obs4>", add_special_tokens=False)["input_ids"][-1]
            obs5 = self.tokenizer("<obs5>", add_special_tokens=False)["input_ids"][-1]
            obs6 = self.tokenizer("<obs6>", add_special_tokens=False)["input_ids"][-1]
            obs7 = self.tokenizer("<obs7>", add_special_tokens=False)["input_ids"][-1]

            static_mask = mask.clone()
            gripper_mask = mask.clone()
            obs_mask = mask.clone()
            static_mask[(text_x != static0) & (text_x != static1)& (text_x != static2)& (text_x != static3) \
                & (text_x != static4) & (text_x != static5)& (text_x != static6)& (text_x != static7) ] = 0
            
            gripper_mask[(text_x != gripper0) & (text_x != gripper1)& (text_x != gripper2)& (text_x != gripper3) \
                & (text_x != gripper4) & (text_x != gripper5)& (text_x != gripper6)& (text_x != gripper7) ] = 0
            
            obs_mask[(text_x != obs0) & (text_x != obs1)& (text_x != obs2)& (text_x != obs3) \
                & (text_x != obs4) & (text_x != obs5)& (text_x != obs6)& (text_x != obs7) ] = 0
            static_mask=static_mask.bool()
            gripper_mask=gripper_mask.bool()
            obs_mask=obs_mask.bool()
            mask=mask.bool()
            action_mask=action_mask.bool()
            if static_mask.sum() < 1 : static_mask = None
            if gripper_mask.sum() < 1 : gripper_mask = None
            if obs_mask.sum() < 1 :  obs_mask = None
            # preimg_mask = mask.clone()
            # preimg_mask[(text_x != preimg0_token_id) & (text_x != preimg1_token_id)& (text_x != preimg2_token_id)& (text_x != preimg3_token_id) \
            #     & (text_x != preimg4_token_id) & (text_x != preimg5_token_id)& (text_x != preimg6_token_id)& (text_x != preimg7_token_id) ] = 0


            if len(self.mask_queue) == 0 and mask is not None:
                self.mask_queue.append(mask)
                for _ in range(self.model.module.pad_length - 1):
                    self.mask_queue.append(mask)
            if len(self.text_queue) == 0 and text_x is not None:
                self.text_queue.append(text_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.text_queue.append(text_x)
            
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                mask = torch.cat(list(self.mask_queue), dim=0)
                text_x = torch.cat(list(self.text_queue), dim=0)
                assert False 
            if self.fusion_mode == 'vit_concat':
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                pass
                assert False 

            if self.use_diff:
                if self.fusion_mode == 'two_way':
                    vision_x = torch.cat([image_x, gripper], dim=0)
                    text_x = text_x.repeat(2, 1)
                    mask = mask.repeat(2, 1)
                    model_out = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor = state, return_feature=True)
                else:
                    model_out = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper = gripper, state_tensor = state, return_feature=True)

                if not get_action:
                    return None
                model_out = model_out.logits
                action_history = torch.tensor(np.stack(self.action_hist_queue, axis=0), dtype=torch.float, device=device).unsqueeze(0)
                action_history = self.normalizer.normalize(action_history)
                if self.supp is None:
                    self.supp = torch.zeros(
                        action_history.shape[0], self.horizon-self.hist_len, action_history.shape[-1], 
                        dtype=action_history.dtype,
                        device=action_history.device,
                    )
                action_history = torch.concat([action_history, self.supp], dim=1)
                act_mask = torch.zeros_like(action_history, device=action_history.device, dtype=torch.bool)
                act_mask[:,:self.hist_len,...] = 1.
                pred_action_seq = self.diffusion_model.conditional_sample(cond_data=action_history, cond_mask=act_mask, global_cond=model_out)
                pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
                action = pred_action_seq[:,self.hist_len:,:]
                if self.future_act_len > 0:
                    action = action[:,:self.future_act_len,:]
                action = action[0]
                action = action.cpu().detach().to(dtype=torch.float16).numpy()
                action[...,-1] = action[...,-1] > 0.5
                action[...,-1] = (action[...,-1] - 0.5) * 2  # scale to -1 or 1
            else:
                if self.fusion_mode == 'two_way':
                    vision_x = torch.cat([image_x, gripper], dim=0)
                    text_x = text_x.repeat(2, 1)
                    mask = mask.repeat(2, 1)
                    action = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor = state, return_feature=True)
                else:
                    action, static_pred, gripper_pred, obs_pred, _ = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper = gripper, state_tensor = state,calib = calib, action_mask = action_mask, static_mask = static_mask, gripper_mask = gripper_mask, obs_mask = obs_mask, return_feature=True)

                if static_pred is not None:
                    obs_preds_rgb_img = static_pred[-1]
                    obs_preds_rgb_img = rearrange(obs_preds_rgb_img, "c h w  -> h w c")
                    obs_preds_rgb_img = np.array(obs_preds_rgb_img.data.cpu())
                #cv2.imwrite("~/liufanfan/workspace/LFF_Lnitial_Fusion_Foundation_Models4/robs_preds_rgb_img.png", ((obs_preds_rgb_img +1) * 128)[:, :, ::-1])
                else:
                    obs_preds_rgb_img = None
                    
                if gripper_pred is not None:
                    obs_preds_gripper_img = gripper_pred[-1]
                    obs_preds_gripper_img = rearrange(obs_preds_gripper_img, "c h w  -> h w c")
                    obs_preds_gripper_img = np.array(obs_preds_gripper_img.data.cpu())
                else:
                    obs_preds_gripper_img = None

                if self.model.module.pad_length != -1:
                    if self.feature_cache is None:
                        self.feature_cache = action.logits[-1]
                    else:
                        new_feat = torch.cat([self.feature_cache[1:], action.logits[-1]], dim=0)
                        self.feature_cache = new_feat
                        if not self.model.module.sep_lm_head:
                            self.model.module.lang_encoder.lm_head.window_size = window_size
                            lm_out = self.model.module.lang_encoder.lm_head(new_feat)
                        else:
                            self.model.module.lm_head.window_size = window_size
                            lm_out = self.model.module.lm_head(new_feat)
                        Output = namedtuple('Output', ['logits'])
                        action = Output(lm_out)

                if 'Temporal' in self.fusion_mode:
                    pose = action.logits[0]
                    gripper = action.logits[1] > 0.5
                    # pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    # gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    if self.args.multi_action_token:
                        pose = pose[:,-1,:]
                        gripper = gripper[:,-1,:]

                    action = torch.cat([pose, gripper], dim=-1)
                    action = action[0] # select 第一个batch的
                else:
                    if self.model.module.act_step == 1:
                        action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2).squeeze(0)[-1] # support multi step history
                    else:
                        pose = action.logits[0]
                        gripper = action.logits[1] > 0.5
                        pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                        gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                        action = torch.cat([pose, gripper], dim=-1)
                        action = action[0] # select first step action
                    
                action[-1] = (action[-1] - 0.5) * 2  # scale to -1 or 1
                action = action.cpu().detach().to(dtype=torch.float16).numpy()
        
        if self.model.module.sep_lm_head:
            self.model.module.lm_head.window_size = window_size
        else:
            self.model.module.lang_encoder.lm_head.window_size = window_size
        if self.model.module.tcp_rel:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state])).unsqueeze(0).float().cpu().detach()
            action = torch.from_numpy(np.stack([action])).unsqueeze(0).float().cpu().detach()
            action = tcp_to_world_frame(action, state)
            action=action.squeeze().to(dtype=torch.float16).numpy()
        return action, None, None, obs_pred


def evaluate_policy(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    if diverse_inst:
        with open('~/yanfeng/project/robotic/RoboFlamingo/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False, num_sequences=0, only_single_task=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    num_sequences = num_sequences if num_sequences>0 else NUM_SEQUENCES
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open('~/yanfeng/project/robotic/RoboFlamingo/lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open('~/yanfeng/project/robotic/RoboFlamingo/eval_sequences.json', 'r') as f:
        eval_sequences_ = json.load(f)
    
    eval_sequences = []
    for e in eval_sequences_:
        if only_single_task and "rotate_blue_block_right" not in e[1][0]: continue # 仅测试使用
        eval_sequences.append(e)
    num_sequences = len(eval_sequences) if num_sequences>len(eval_sequences) else num_sequences
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    # assert num_sequences % device_num == 0
    # interval_len = int(num_sequences // device_num); print("**********YF: int(num_sequences // device_num))*****************")
    interval_len = int((num_sequences+device_num-1) // device_num)
    while len(eval_sequences)<device_num*interval_len:
        eval_sequences += eval_sequences[:(device_num*interval_len-len(eval_sequences))]
    # print(eval_sequences)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, len(eval_sequences))]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    # if not debug:
    #     eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                f"{device_id}/{device_num}|"+" ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        local_sequence_i += 1
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    print("world_size:%d, rank:%d"%(torch.distributed.get_world_size(), torch.distributed.get_rank()))
    
    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    # 确保所有进程同步
    # torch.cuda.set_device(args.device)
    torch.cuda.synchronize()  # 同步 GPU
    torch.distributed.barrier()
    print(f"{torch.distributed.get_rank()} end!!!")
    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    if torch.distributed.get_rank() == 0:
        all_res_tup = [None for _ in range(device_num)] 
    else:
        all_res_tup = None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)
    print(f"{torch.distributed.get_rank()} gather_object end!!!")
    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list = res_tup_list[:num_sequences]
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)
        print(f"calvin succeed", np.array(results).mean())
        sys.stdout.flush()  # 刷新缓冲区，确保输出即时被捕获
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, robot_obs=robot_obs, scene_obs=scene_obs, diverse_inst=diverse_inst)
        else:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i,diverse_inst=diverse_inst)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset()
    start_info = env.get_info()

    debug = True
    if debug:
        img_queue = []
        img_queue2 = []

    for step in range(EP_LEN):
        if model.replan != -1 and step % model.replan == 0:
            if model.model.module.refresh != -1:
                model.model.module.lang_encoder.lm_head.hidden_state = None
                model.model.module.lang_encoder.lm_head.history_memory = model.model.module.lang_encoder.lm_head.history_memory[-model.refresh:]
            else:
                model.reset()
        action, obs_preds_rgb_img, obs_preds_gripper_img, obs_pred = model.step(obs, lang_annotation,env, (len(planned_actions) == 0))

        if obs_pred is not None:
            voxel_range = [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]
            voxel_size = [0.025, 0.025, 0.025*2]
            vfe_generator = OccupancyVFE(voxel_range, voxel_size)
            # occ = model.model.module.occ #(bs, w, h, z, c)
            obs_pred[0][:,:,:,0] = torch.sigmoid(obs_pred[0][:,:,:,0])
            obs_pred = np.array(obs_pred[0].cpu())
            point,rgb = vfe_generator.decode_occupied_grid(obs_pred)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            

            # pcd_dir = os.path.join(eval_log_dir, 'pcd')
            
            #img_clip.write_gif(os.path.join(val_dir, f'{success_counter}-{task_seq}.gif'), fps=30)
            pcd_dir = os.path.join(eval_log_dir, f'pcd/{sequence_i}-{subtask_i}-{subtask}')

            if not os.path.exists(pcd_dir):
                try:
                    os.makedirs(pcd_dir)
                except:
                    print('dirs_maked')


            pcd_path = os.path.join(pcd_dir, f'pcd_{step}.pcd')

            o3d.io.write_point_cloud(pcd_path, pcd)

        # s_dir = os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}')
        # s_path = os.path.join(s_dir, f'pcd_{step}.pcd')
        # print(s_path)


        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        if model.use_diff:
            model.action_hist_queue.append(action)
        obs, _, _, current_info = env.step(action)
        if debug:
            
            
            obs_preds_rgb_img, obs_preds_gripper_img
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_copy_gr = copy.deepcopy(obs['rgb_obs']['rgb_gripper'])
            img_copy = cv2.resize(img_copy, (200,200))
            img_copy_gr = cv2.resize(img_copy_gr, (200,200))
            img_copy = cv2.hconcat((img_copy, img_copy_gr))
            if (obs_preds_rgb_img is not None)  and (obs_preds_gripper_img is not None):
                obs_preds_rgb_img =  cv2.resize(obs_preds_rgb_img,(200,200))
                obs_preds_gripper_img = cv2.resize(obs_preds_gripper_img,(200,200))

            

                obs_preds_rgb_img = cv2.hconcat((obs_preds_rgb_img, obs_preds_gripper_img))

                obs_preds_rgb_img = ((obs_preds_rgb_img*(0.26862954, 0.26130258, 0.27577711))+(0.48145466, 0.4578275, 0.40821073))*255

                obs_preds_rgb_img  = np.clip(obs_preds_rgb_img, 0, 255)

                obs_preds_rgb_img = obs_preds_rgb_img.astype(np.uint8)
                img_copy = cv2.vconcat((img_copy, obs_preds_rgb_img))

          
          
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                eval_dir = os.path.join(eval_log_dir, 'eval')
                try:
                    os.makedirs(eval_dir)
                except:
                    print('dirs_maked')
                    
                img_clip = ImageSequenceClip(img_queue, fps=30)
                img_clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30) # , logger=None
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        
        eval_dir = os.path.join(eval_log_dir, 'eval')
        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except:
                print('dirs_maked')
           
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30) # , logger=None
    return False

def eval_one_epoch_calvin(args, model, dataset_path, image_processor, tokenizer, future_act_len=-1):

    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    wrapped_model = ModelWrapper(args, model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=args.n_obs_steps, future_act_len=future_act_len)
    evaluate_policy(wrapped_model, env, 0, args.calvin_conf_path)


def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False, num_sequences=0):

    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type=="diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    wrapped_model = ModelWrapper(args ,model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=hist_len, future_act_len=future_act_len)
    evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst, num_sequences=num_sequences, only_single_task=args.only_single_task)


def eval_one_epoch_calvin_with_dataloder(args, model, calvin_loader, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False):
    
    total_training_steps = calvin_loader.num_batches
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=0,
    )
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = hist_len = args.n_obs_steps if args.head_type=="diffusion" else args.pad_length if args.pad_length != -1 else None
    wrapped_model = ModelWrapper(args, model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=hist_len, future_act_len=future_act_len)

    plans = defaultdict(list)
    results = []
    for num_steps, batch_calvin in t:
        assert len(batch_calvin) == 1
        batch_calvin = batch_calvin[0]  # 先假设只有一个batch
        if wrapped_model.model.module.env is None:
            dataset_path=batch_calvin['dataset_path']
            wrapped_model.model.module.env = get_env(Path(dataset_path), show_gui=False)  # make_env(dataset_path)
        batch_calvin.update({"env": wrapped_model.model.module.env})
        generator = batch_calvin.pop('generator')(**batch_calvin)
        
        wrapped_model.reset()

        debug = True
        img_queue = []
        planned_actions = []
        dt = next(generator)
        while not dt['done']:
            
        
            step = dt["step_cur"]
            if wrapped_model.replan != -1 and step % wrapped_model.replan == 0:
                if wrapped_model.model.module.refresh != -1:
                    wrapped_model.model.module.lang_encoder.lm_head.hidden_state = None
                    wrapped_model.model.module.lang_encoder.lm_head.history_memory = wrapped_model.model.module.lang_encoder.lm_head.history_memory[-wrapped_model.refresh:]
                else:
                    wrapped_model.reset()
                if dt["is_reset"]: 
                    wrapped_model.reset()

            lang_annotation = dt["lang"]
            obs={
                "rgb_obs": {
                    "rgb_static": dt['rgb_static_ori'],
                    "rgb_gripper": dt['rgb_gripper_ori'],
                },
                "robot_obs": dt['robot_obs'],
            }
            action, obs_preds_rgb_img, obs_preds_gripper_img, obs_pred = wrapped_model.step(obs, lang_annotation, wrapped_model.model.module.env, (len(planned_actions) == 0))

            task_seq = f','.join(dt['eval_sequence'])
            if len(planned_actions) == 0:
                if action.shape == (7,):
                    planned_actions.append(action)
                else:
                    planned_actions.extend([action[i] for i in range(action.shape[0])])
            action = planned_actions.pop(0)
            if wrapped_model.use_diff:
                wrapped_model.action_hist_queue.append(action)
            if step == 0:
                # for tsne plot, only if available
                subtask = dt["eval_sequence"][dt["subtask_i"]]
                collect_plan(wrapped_model, plans, subtask)

            dt = generator.send(action)
            success_counter = dt['success_counter']

            val_dir = os.path.join(eval_log_dir, 'val')
            if not os.path.exists(val_dir):
                try:
                    os.makedirs(val_dir)
                except:
                    print('dirs_maked')
            if debug:
                
                img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
                img_copy_gr = copy.deepcopy(obs['rgb_obs']['rgb_gripper'])
                
                img_copy_gr = cv2.resize(img_copy_gr, (200,200))
                img_copy = cv2.hconcat((img_copy, img_copy_gr))
                if (obs_preds_rgb_img is not None)  and (obs_preds_gripper_img is not None):
                    obs_preds_rgb_img =  cv2.resize(obs_preds_rgb_img,(200,200))
                    obs_preds_gripper_img = cv2.resize(obs_preds_gripper_img,(200,200))

                

                    obs_preds_rgb_img = cv2.hconcat((obs_preds_rgb_img, obs_preds_gripper_img))

                    obs_preds_rgb_img = ((obs_preds_rgb_img*(0.26862954, 0.26130258, 0.27577711))+(0.48145466, 0.4578275, 0.40821073))*255

                    obs_preds_rgb_img  = np.clip(obs_preds_rgb_img, 0, 255)

                    obs_preds_rgb_img = obs_preds_rgb_img.astype(np.uint8)
                    img_copy = cv2.vconcat((img_copy, obs_preds_rgb_img))

                
                img_queue.append(img_copy)

                if obs_pred is not None:
                    voxel_range = [[-0.5, 0.5], [-0.5, 0.5], [0.3, 0.8]]
                    voxel_size = [0.025, 0.025, 0.025]
                    vfe_generator = OccupancyVFE(voxel_range, voxel_size)
                    # occ = model.model.module.occ #(bs, w, h, z, c)
                    obs_pred[0][:,:,:,0] = torch.sigmoid(obs_pred[0][:,:,:,0])
                    obs_pred = np.array(obs_pred[0].cpu())
                    point,rgb = vfe_generator.decode_occupied_grid(obs_pred)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(rgb)
                    # pcd_dir = os.path.join(eval_log_dir, 'pcd')
                    
                    #img_clip.write_gif(os.path.join(val_dir, f'{success_counter}-{task_seq}.gif'), fps=30)
                    pcd_dir = os.path.join(val_dir, f'pcd/{success_counter}-{task_seq}')
                    if not os.path.exists(pcd_dir):
                        try:
                            os.makedirs(pcd_dir)
                        except:
                            print('dirs_maked')
                    pcd_path = os.path.join(pcd_dir, f'pcd_{step}.pcd')
                    o3d.io.write_point_cloud(pcd_path, pcd)

         

        
        if debug:
            # print(colored("success", "green"), end=" ")
            img_clip = ImageSequenceClip(img_queue, fps=30)
            task_seq = f','.join(dt['eval_sequence'])

            val_dir = os.path.join(eval_log_dir, 'val')
            if not os.path.exists(val_dir):
                try:
                    os.makedirs(val_dir)
                except:
                    print('dirs_maked')
            img_clip.write_gif(os.path.join(val_dir, f'{success_counter}-{task_seq}.gif'), fps=30) # , logger=None
            logging.info(os.path.join(eval_log_dir, f'{success_counter}-{task_seq}.gif'))
            
        results.extend([success_counter])
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True
    
    if is_dist_avail_and_initialized():
        dist.barrier()

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp
    
    device_num = int(torch.distributed.get_world_size())
    all_res_tup = [copy.deepcopy(results) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(results, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        results = merge_multi_list(all_res_tup)
    
    return results


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = checkpoint.stem.split("=")[1]
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


def generate_zero_shot_instr():
    random.seed(123)
    with open('~/yanfeng/project/robotic/RoboFlamingo/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    eval_sequences = get_sequences(NUM_SEQUENCES)
    
    all_res = []
    for initial_state, eval_sequence in eval_sequences:
        res = []
        for subtask_i, subtask in enumerate(eval_sequence):
            res.append(random.choice(val_annotations[subtask]))
        all_res.append(res)
    with open('~/yanfeng/project/robotic/RoboFlamingo/lang_annotation_cache.json', 'w') as f:
        json.dump(all_res, f, indent=1)


def save_sequences():
    random.seed(123)
    eval_sequences = get_sequences(NUM_SEQUENCES)
    with open('~/yanfeng/project/robotic/RoboFlamingo/eval_sequences.json', 'w') as f:
        json.dump(eval_sequences, f)



if __name__ == "__main__":
    main()  