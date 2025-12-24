import argparse
from collections import Counter, defaultdict, namedtuple
import logging
import os, json, random, pickle
from pathlib import Path
import sys, gc
import time
import h5py
import PIL.Image as Image
import copy
from collections import deque
from itertools import chain
from typing import Tuple, Dict, List, Set, Union, Any, Optional, Mapping, cast, Literal, Sequence, Iterable
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
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
logger = logging.getLogger(__name__)

EP_LEN = 600 # 源代码中最大值设置为600
resolution = (224, 224)
NUM_SEQUENCES = 1000
# NUM_SEQUENCES = 400
import pybullet as pb
import cv2
from matplotlib import pyplot as plt


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
    def __init__(self, args, model, tokenizer, image_processor, cast_dtype, use_diff, history_len=None, future_act_len=-1):
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
        self.global_his_img_queue = []
        self.global_his_pos_queue = []
        self.his_vision_queue = deque(maxlen=history_len)
        self.his_pose_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.calib_queue = deque(maxlen=history_len)

        self.nav_history_len = args.nav_history_len
        print(f"nav_history_len: {args.nav_history_len}")

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
        self.global_his_img_queue = []
        self.global_his_pos_queue = []
        self.his_vision_queue = deque(maxlen=history_len)
        self.his_pose_queue = deque(maxlen=history_len)
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

    def _read_history(self, global_step, nav_history_len=20, distance_threshold=0.1, SS=0.95):
        def _repeat_frame(matrix_a, matrix_b, threshold=0.1):
            """
            判断matrix_a与matrix_b中哪些帧的欧氏距离小于threshold，返回这些帧的索引
            """
            matrix_a = np.array(matrix_a)   # shape: (3,)
            matrix_b = np.array(matrix_b)   # shape: (l, 3)
            distances = np.linalg.norm(matrix_b - matrix_a, axis=1)
            repeat_indices = np.where(np.abs(distances) < threshold)[0]
            return repeat_indices
        
        def _similar_frame(vision_gripper_pool, vision_gripper_list, indices, SS):
            """
            判断vision_gripper_pool与vision_gripper_list中indices指定的帧是否有相似度大于SS的
            """
            if len(indices) == 0:
                return False
            
            # 将CUDA张量转换为NumPy数组
            vision_gripper_pool_np = vision_gripper_pool.cpu().numpy()
            vision_gripper_list_np = [item.cpu().numpy() for item in vision_gripper_list]
            vision_gripper_list_np = np.array(vision_gripper_list_np)
            
            # 调整形状以便计算余弦相似度
            selected = vision_gripper_list_np[indices]  # 形状: (m, 1, 1024)
            # 判断selected的形状是否为3维，如果是则将前两个维度合并
            if len(selected.shape) == 3:
                selected = selected.reshape(-1, selected.shape[-1])
                
            v = vision_gripper_pool_np.reshape(1, -1)  # 形状: (1, 1024)
            
            # 计算余弦相似度
            dot_products = np.dot(selected, v.T).flatten()
            norms_list = np.linalg.norm(selected, axis=1)
            norm_v = np.linalg.norm(vision_gripper_pool_np)
            cosine_similarities = dot_products / (norms_list * norm_v + 1e-8)
            
            return np.any(cosine_similarities > SS)

        frame_id = global_step

        if frame_id < nav_history_len:
            history_list = [0] * (nav_history_len - frame_id - 1) + list(range(frame_id + 1))
        else:
            history_list = list(range(frame_id - nav_history_len + 1, frame_id + 1))
        history_list.reverse()

        global_pose = None
        past_frame_id = None
        vision_gripper_list, pose_list = [], []
        device = self.global_his_img_queue[-1].device
        for local_frame_id in history_list:
            try:
                pose = self.global_his_pos_queue[local_frame_id]
                vision_gripper_pool = self.global_his_img_queue[local_frame_id] # 这里用tensor，最后返回torch.stack
            except Exception as e:
                print(f"read {local_frame_id} in global_his_pos_queue or global_his_img_queue failed: {e}")
                return None, None
                
            if local_frame_id == frame_id: # 第一帧原点不用计算，直接保存。当frame_id=0的时候一直在这
                global_pose = pose
                local_pose = pose - global_pose
                vision_gripper_list.append(vision_gripper_pool)
                pose_list.append(local_pose) # np.array
            else:
                if past_frame_id == 0 and local_frame_id == 0: # 当剩下的都是0，直接保存
                    vision_gripper_list.append(vision_gripper_list[-1])
                    pose_list.append(pose_list[-1])
                else:
                    local_pose = pose - global_pose

                    repeat_indices = _repeat_frame(local_pose, pose_list, threshold=distance_threshold)
                    if not _similar_frame(vision_gripper_pool, vision_gripper_list, repeat_indices, SS):
                        vision_gripper_list.append(vision_gripper_pool)
                        pose_list.append(local_pose)
                    else:
                        new_frame = 0 if history_list[-1] == 0 else history_list[-1] - 1
                        history_list.append(new_frame)

            past_frame_id = local_frame_id

            if len(vision_gripper_list) == nav_history_len and len(pose_list) == nav_history_len:
                break

        vision_gripper_ts = torch.cat(vision_gripper_list, dim=0).unsqueeze(0).unsqueeze(0).to(device) # 1 1 nav_history_len 1024
        pose_ts = torch.from_numpy(np.array(pose_list)).unsqueeze(0).unsqueeze(0).to(device) # 1 1 nav_history_len 3

        return vision_gripper_ts, pose_ts

    def step(self, obs, goal, env, get_action=True, global_step=0):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        from scipy.spatial.transform import Rotation as R
        def calculate_extrinsic_matrix(position, rotation):
            # 将欧拉角转换为旋转矩阵
            if len(rotation) == 3: # 假设输入的是欧拉角 (pitch, yaw, roll)的角度表示
                rotation_matrix = R.from_euler('xyz', rotation, True).as_matrix()
            elif len(rotation) == 4:  # 假设输入的是四元数
                rotation_matrix = R.from_quat(rotation).as_matrix()
            else:
                raise ValueError("Rotation should be either 3 (Euler angles) or 4 (Quaternion) elements.")
            extrinsic_matrix = np.eye(4) # 构建4x4外参矩阵
            extrinsic_matrix[:3, :3] = rotation_matrix
            extrinsic_matrix[:3, 3] = position
            return extrinsic_matrix
        def calculate_intrinsic_matrix(fov_x_deg, width, height):
            fov_y_deg = fov_x_deg
            fov_x = np.deg2rad(fov_x_deg) # 将角度转换为弧度
            fov_y = np.deg2rad(fov_y_deg)
            f_x = width / (2 * np.tan(fov_x / 2)) # 计算焦距
            f_y = height / (2 * np.tan(fov_y / 2))
            c_x = width / 2 # 计算光心位置
            c_y = height / 2
            intrinsic_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y],[0, 0, 1]]) # 构建内参矩阵
            return intrinsic_matrix

        obs["rgb_obs"]['rgb_static'] = obs["rgb_obs"]['rgb_static'][:, 86:-86]
        obs["rgb_obs"]['rgb_gripper'] = obs["rgb_obs"]['rgb_gripper'][:, 86:-86]

        base_pos = list(env.last_event.metadata["agent"]["position"].values()) # [2.25, 0.9009921550750732, 7.25]
        base_pos_new = [base_pos[0], -base_pos[1], base_pos[2]]
        base_rot = list(env.last_event.metadata["agent"]["rotation"].values()) # [-0.0, 0.0, 0.0]
        base_rot_new = [-base_rot[0], base_rot[1], base_rot[2]]
        base_rot = base_pos_new + base_rot_new

        calib = {}
        static_pos = list(env.last_event.metadata["cameraPosition"].values()) # [2.251920223236084, 1.445693016052246, 7.317880630493164]
        static_pos_new = [static_pos[0], -static_pos[1], static_pos[2]] # [2.251920223236084, -1.445693016052246, 7.317880630493164]
        static_rot = list(dict(x=env.last_event.metadata["agent"]["cameraHorizon"], y=base_rot[4], z=base_rot[5]).values()) # [30.000059127807617, 0.0, 0.0]
        static_rot_new = [-static_rot[0], static_rot[1], static_rot[2]] # [-30.000059127807617, 0.0, 0.0]
        static_extrinsic = calculate_extrinsic_matrix(static_pos_new, static_rot_new)
        static_fov = env.last_event.metadata["fov"] # 59.0
        static_intrinsic = calculate_intrinsic_matrix(static_fov, obs["rgb_obs"]['rgb_static'].shape[1], obs["rgb_obs"]['rgb_static'].shape[0])
        cam_config = {'fov': static_fov,  'width': obs["rgb_obs"]['rgb_static'].shape[1], 'height': obs["rgb_obs"]['rgb_static'].shape[0]}
        calib["rgb_static"] = { 'extrinsic_matrix':static_extrinsic,
                                'intrinsic_matrix':static_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                'cam_config': cam_config}

        gripper_pos = list(env.last_event.metadata["thirdPartyCameras"][0]["position"].values()) # array([2.34115124, 1.42482579, 1.1460948 ])
        gripper_pos_new = [gripper_pos[0], -gripper_pos[1], gripper_pos[2]] # pose的y轴取负！！！
        gripper_rot = list(env.last_event.metadata["thirdPartyCameras"][0]["rotation"].values()) # array([ 2.99999905e+01,  1.80000000e+02, -7.47154218e-07])
        gripper_rot_new = [-gripper_rot[0], gripper_rot[1], gripper_rot[2]] # [-29.999990463256836, 180.0, -7.47154217606294e-07] # rota的x轴取负！！！
        gripper_extrinsic = calculate_extrinsic_matrix(gripper_pos_new, gripper_rot_new)
        gripper_fov = env.last_event.metadata["thirdPartyCameras"][0]["fieldOfView"]
        gripper_intrinsic = calculate_intrinsic_matrix(gripper_fov, obs["rgb_obs"]['rgb_gripper'].shape[1], obs["rgb_obs"]['rgb_gripper'].shape[0])

        cam_config = {'fov': gripper_fov, 'width': obs["rgb_obs"]['rgb_gripper'].shape[1], 'height': obs["rgb_obs"]['rgb_gripper'].shape[0],}
        calib["rgb_gripper"] = {'extrinsic_matrix':gripper_extrinsic,
                                'intrinsic_matrix':gripper_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                'cam_config': cam_config}

        # cam外参矩阵YZ*负号与deproject函数对齐
        static_extrinsic_matrix = np.linalg.inv(calib['rgb_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        gripper_extrinsic_matrix = np.linalg.inv(calib['rgb_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        base_pose = calculate_extrinsic_matrix(base_rot[:3], base_rot[3:])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_pose)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_pose)
        # cam外参矩阵YZ*T矩阵与calvin点云坐标对齐
        T_translate = np.array([
            [1, 0, 0, 0.0],    # 
            [0, 1, 0, 1.0],     # 
            [0, 0, 1, 0.0],    #  
            [0, 0, 0, 1] 
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        base_pose = np.dot(base_pose, T_translate)
        R = np.array([
            [1, 0, 0, 0],
            [0,  0, -1, 0],
            [0,  1, 0, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往右X轴；往下为Y轴；往前为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        base_pose = np.dot(base_pose, R)
        static_pose = base_pose[:3, 3]
        if 0:
            from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam
            static_cam = cam(static_extrinsic_matrix, calib['rgb_static']['cam_config']['height'], calib['rgb_static']['cam_config']['width'], calib['rgb_static']['cam_config']['fov'])
            gripper_cam = cam(gripper_extrinsic_matrix, calib['rgb_gripper']['cam_config']['height'], calib['rgb_gripper']['cam_config']['width'], calib['rgb_gripper']['cam_config']['fov'])

            rgb_static, depth_static = obs["rgb_obs"]['rgb_static'], env.last_event.depth_frame[:, 86:-86]
            rgb_gripper, depth_gripper = obs["rgb_obs"]['rgb_gripper'],  env.last_event.third_party_depth_frames[0][:, 86:-86]

            static_pcd = deproject(static_cam, depth_static, homogeneous=False, sanity_check=False).transpose(1, 0)
            gripper_pcd = deproject(gripper_cam, depth_gripper, homogeneous=False, sanity_check=False).transpose(1, 0)
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
        image = obs["rgb_obs"]['rgb_static'] # (224, 224, 3)
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image]) # torch.Size([1, 3, 224, 224])
        # expand image dimension 缺少T维度
        image_x = image_x.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type) # torch.Size([1, 1, 1, 3, 224, 224])
        
        # fix window_size : ddp_model -> ... -> window_size
        if self.model.module.sep_lm_head:
            window_size = self.model.module.lm_head.window_size # zyf 第一次的时候window_size=12吗？
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
            gripper = obs["rgb_obs"]['rgb_gripper'] # (224, 224, 3)
            gripper = Image.fromarray(gripper)
            gripper = self.image_process_fn([gripper]) # torch.Size([1, 3, 224, 224])
            # expand image dimension
            gripper = gripper.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type) # torch.Size([1, 1, 1, 3, 224, 224])
        
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
                self.model.module.pad_length = self.model.module.window_size # zyf 这里赋值12

            self.model.module.pad_length = -1   # zyf 为什么这里强制赋值-1？
            # if self.model.module.pad_length != -1:
            if len(self.img_queue) == 0: # 增加T维度，12帧
                self.img_queue.append(image_x) # zyf 一开始只有1帧
                for _ in range(self.model.module.pad_length - 1): # zyf pad_length=-1，-1-1=-2
                    self.img_queue.append(image_x)
            else:
                self.img_queue.append(image_x) # zyf 逐渐增加
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

            self.global_his_pos_queue.append(static_pose)
            assert self.args.use_static == False, "In navigation dataset, self.args.use_static must be False!!!"
            vit_feat = self.model(vision_x=image_x, lang_x=None, vision_gripper=gripper, extract_feat_test=True, use_static=self.args.use_static) # 1 1024 tensor
            self.global_his_img_queue.append(vit_feat)

            his_vision, his_pose = self._read_history(global_step, self.nav_history_len) # 1 1 20 c tensor
            
            self.his_vision_queue.append(his_vision)
            self.his_pose_queue.append(his_pose)

            if 'Temporal' in self.fusion_mode: # zyf 这里生成T维度，下一步补齐维度，这里用stack更好一些
                image_x = torch.cat(list(self.img_queue), dim=1) # torch.Size([1, T, 1, 3, 224, 224])
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=1) # torch.Size([1, T, 1, 3, 224, 224])
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=1)
                if his_vision is not None:
                    his_vision = torch.cat(list(self.his_vision_queue), dim=1) # 1 t 20 1024
                if his_pose is not None:
                    his_pose = torch.cat(list(self.his_pose_queue), dim=1) # 1 t 20 3

                image_x = image_x.unsqueeze(2) # torch.Size([1, T, 1, 1, 3, 224, 224])
                gripper = gripper.unsqueeze(2) # torch.Size([1, T, 1, 1, 3, 224, 224])
                calib = [torch.cat([tmp[i_calibe] for tmp in self.calib_queue], dim=1) for i_calibe in range(len(self.calib_queue[0]))] # 9 T 将每一项（len(self.calib_queue[0])）都按照帧顺序（dim=1）拼接

            self.model.module.pad_length = -1

            # expand text dimension
            text_x, mask = self.text_process_fn([goal], window_size=len(self.img_queue)) # torch.Size([1, m+T])，看preprocess_text_calvin函数
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

            # 这两个queue并不使用
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
                    action, static_pred, gripper_pred, obs_pred, _ = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper = gripper, \
                        state_tensor = state, calib = calib, action_mask = action_mask, static_mask = static_mask, gripper_mask = gripper_mask, obs_mask = obs_mask, \
                        return_feature=True, his_vision_static = his_vision, his_pose = his_pose, use_static=self.args.use_static)

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
                    pose = action.logits[0] # torch.Size([1, T, 6])
                    gripper = action.logits[1].argmax(-1)[..., None] # torch.Size([1, T, 20]) -> torch.Size([1, T]) -> torch.Size([1, T, 1])
                    # pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    # gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    if self.args.multi_action_token: # 选择最后一帧作为结果，符合因果解码
                        pose = pose[:,-1,:] # torch.Size([1, 6])
                        gripper = gripper[:,-1,:] # torch.Size([1, 1])

                    action = torch.cat([pose, gripper], dim=-1) # torch.Size([1, 7])
                    action = action[0] # select 第一个batch的 torch.Size([7])
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
                    
                # action[-1] = (action[-1] - 0.5) * 2  # scale to -1 or 1
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


def evaluate_policy_ddp(model, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False, args=None, only_single_task=False, resume=False):
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
    os.environ['OBJAVERSE_HOUSES_DIR'] = os.environ.get(
        'OBJAVERSE_HOUSES_DIR',
        "~/zhongyufeng/code/spoc-robot-training/downloads/objaverse_houses/houses_2023_07_28"
    )

    os.environ['OBJAVERSE_DATA_DIR'] = os.environ.get(
        'OBJAVERSE_DATA_DIR',
        "~/zhongyufeng/code/spoc-robot-training/downloads/objaverse_assets/2023_07_28"
    )
    path_to_adds = ["~/zhongyufeng/code/robotron-nav/third_party", "~/zhongyufeng/code/robotron-nav/third_party/spoc_robot_training"]
    import glob
    from datetime import datetime

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    def get_subdirectories_sorted_by_creation_time(f_dir):
        subdirectories = [os.path.join(f_dir, item) for item in os.listdir(f_dir) if os.path.isdir(os.path.join(f_dir, item))]
        subdirectories.sort(key=lambda x: os.path.getctime(x))
        return subdirectories

    if rank == 0:
        # Only rank 0 process generates the flag
        if not os.path.exists(eval_log_dir):
            os.makedirs(eval_log_dir)
        else:
            print(f"{eval_log_dir} is existed")

        created_dir = get_subdirectories_sorted_by_creation_time(eval_log_dir)
        if resume and len(created_dir) != 0:
            flag = created_dir[-1].split('/')[-1]
        else:
            flag = datetime.now().strftime('%m-%d-%H-%M-%S')

        flag_tensor = torch.tensor([ord(c) for c in flag], dtype=torch.int, device=rank)
    else:
        # Other ranks use a placeholder tensor
        flag_tensor = torch.empty(14, dtype=torch.int, device=rank)  # Assuming the flag length is 14 characters

    # 打印广播前的张量
    # print(f"Rank {rank} before broadcast: {flag_tensor}")

    # Broadcast the flag tensor from rank 0 to all other ranks
    dist.broadcast(flag_tensor, src=0)

    # Convert the tensor back to string
    flag = ''.join([chr(c.item()) for c in flag_tensor])
    # print(f"Rank {rank} after broadcast: {flag}")

    print("world_size:%d, rank:%d, flag:%s"%(torch.distributed.get_world_size(), torch.distributed.get_rank(), flag))

    def extract_house_list(eval_log_dir, eval_sequences, flag):
        # 查找所有gif文件
        eval_dir = os.path.join(eval_log_dir, flag)
        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except:
                print('dirs_maked')

        gif_files = glob.glob(os.path.join(eval_dir, "*.gif"))
        
        house_list = []

        for gif_file in gif_files:
            # 提取文件名（不带路径）
            file_name = os.path.basename(gif_file)
            # 去掉文件的后缀
            file_name_no_ext = os.path.splitext(file_name)[0]
            # 按照'-'字符分割文件名，并取第一个字符
            house_name = file_name_no_ext.split('_')[0]
            # 添加到house_list
            house_list.append(house_name)

        filtered_eval_sequences = []
        for i, seq in enumerate(eval_sequences):
            seq["sequence_i"] = i
            if str(i) not in house_list:
                # print(i)
                filtered_eval_sequences.append(seq)

        return filtered_eval_sequences

    # try:
    for path_to_add in path_to_adds:
        if path_to_add not in sys.path:
            sys.path.append(path_to_add) 

    import prior
    import ai2thor.platform
    from ai2thor.controller import Controller
    from spoc_robot_training.utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
    from spoc_robot_training.utils.task_type_mapping_utils import inverse_map_task_type
    STRETCH_ENV_ARGS["renderInstanceSegmentation"] = True # zyf 在nav的测试中需要
    STRETCH_ENV_ARGS["renderDepthImage"] = True
    OBJAVERSE_HOUSES_DIR = os.environ['OBJAVERSE_HOUSES_DIR']
    special_task = args.calvin_dataset[0].split('/')[-2] # "ObjectNavRoom" # zyf 需要特别指定任务！！！，同spoc-robot-training/training/offline/dataset_mixtures.py 
    # sample 房间id和机器人位置
    for _ in range(10):
        try:
            EVAL_TASKS = prior.load_dataset(
                dataset="spoc-data",
                entity="spoc-robot",
                revision="chores-small",
                task_types=[inverse_map_task_type(special_task)],
            )
            samples = EVAL_TASKS["val"] # 像字典一样读取
            break
        except Exception as e:
            print(f"Error loading scenes: {e}")
            continue

    eval_sequences = []
    special_house_id = None
    for task in samples:
        eval_sequences.append(task)
    eval_loop_num=1

    # torch.distributed.barrier() # 防止计数不一致
    eval_sequences = extract_house_list(eval_log_dir, eval_sequences, flag)
    eval_sequences = eval_sequences
    n_tasks = len(eval_sequences)
    eval_log_dir = get_log_dir(eval_log_dir)

    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    interval_len = int((len(eval_sequences) + device_num - 1) // device_num) # 计算每个设备需要处理的任务数量，+ device_num - 1是为了向上取整，比如8//3=2，变成（8+2）//3=3
    while len(eval_sequences) < device_num * interval_len: # 如果 eval_sequences 的数量不足以平均分配给所有设备，则重复已有的任务，直到任务数量达到 device_num * interval_len
        eval_sequences += eval_sequences[:(device_num * interval_len - len(eval_sequences))] # 这里的device_num * interval_len - len(eval_sequences)，就是3*3-8
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, len(eval_sequences))] # 每个设备根据其 device_id 从 eval_sequences 中获取对应的任务子集

    print("world_size:%d, rank:%d, eval_sequences_num:%d"%(torch.distributed.get_world_size(), torch.distributed.get_rank(), len(eval_sequences)))

    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len # 计数idx是与每个设备根据的device_id相关
    
    # 房间assets
    subset = "val"
    max_houses_per_split = {subset: int(1e9)}
    for _ in range(10):
        try:
            datasets = prior.load_dataset(
                dataset="spoc-data",
                entity="spoc-robot",
                revision="local-objaverse-procthor-houses",
                path_to_splits=None,
                split_to_path={k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz") for k in max_houses_per_split.keys()},
                max_houses_per_split=max_houses_per_split,
            )
            break
        except Exception as e:
            print(f"Error loading assets: {e}")
            continue
    # 仿真器初始化
    for _ in range(10):
        try:
            controller_args={
                **STRETCH_ENV_ARGS,
                "platform": (
                    ai2thor.platform.OSXIntel64 if sys.platform.lower() == "darwin" else ai2thor.platform.CloudRendering
                ),
            }
            env = Controller(**controller_args)
            break
        except Exception as e:
            print(f"Error init env: {e}")
            continue      
            
    for task in eval_sequences:
        try:
            result = evaluate_sequence(env, model, eval_loop_num, datasets[subset], task, plans, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst, flag=flag)
            results.append(result)
            local_sequence_i += 1
        except Exception as e:
            print(f"Error eval task: {e}")
            for _ in range(3):
                try:
                    controller_args={
                        **STRETCH_ENV_ARGS,
                        "platform": (
                            ai2thor.platform.OSXIntel64 if sys.platform.lower() == "darwin" else ai2thor.platform.CloudRendering
                        ),
                    }
                    env = Controller(**controller_args)
                    break
                except Exception as e:
                    print(f"Error init env: {e}")
                    continue  
            continue
            
    # finally:
    #     [sys.path.remove(path_to_add) for path_to_add in path_to_adds if path_to_add in sys.path]
   
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    print("world_size:%d, rank:%d, finished"%(torch.distributed.get_world_size(), torch.distributed.get_rank()))
    
    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    def calculate_succ_rate_and_average_score_and_percentage_room(dir_path):
        # 查找所有gif文件
        gif_files = glob.glob(os.path.join(dir_path, "*.gif"))
        total_files = len(gif_files)  # 总文件数量
        num_succ = 0  # 统计含有'succ'的文件数量
        total_score = 0.0  # 所有成功文件的分数之和
        total_room = 0.0

        for gif_file in gif_files:
            file_name = os.path.basename(gif_file)
            is_succ = file_name.split('.gif')[0].split('_')[-3]
            if is_succ == 'succ':
                num_succ += 1
                # 提取并累加分数
                try:
                    score_str = file_name.split('.gif')[0].split('_')[-2]  # 假设分数总是倒数第二个元素
                    score = float(score_str)
                    total_score += score
                except ValueError:
                    print(f"Warning: Score extraction failed for {file_name}")

            room_str = file_name.split('.gif')[0].split('_')[-1]
            room = float(room_str)
            total_room += room

        # 计算含有'succ'的文件比例
        succ_rate = num_succ / total_files if total_files > 0 else 0.0
        # 计算平均分数
        average_score = total_score / total_files if total_files > 0 else 0.0
        percentage_room = total_room / total_files if total_files > 0 else 0.0

        print(f"Total number of GIF files: {total_files}")
        print(f"Success rate: {succ_rate:.4f}")
        print(f"Average SEL: {average_score:.4f}")
        print(f"Percentage room visited: {percentage_room:.4f}")

        return succ_rate, average_score, percentage_room
    
    # 确保所有进程同步
    # torch.cuda.set_device(args.device)
    torch.cuda.synchronize()  # 同步 GPU
    torch.distributed.barrier()
    print(f"rank: {torch.distributed.get_rank()} end!!!")
    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    if torch.distributed.get_rank() == 0:
        all_res_tup = [None for _ in range(device_num)] 
    else:
        all_res_tup = None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)
    print(f"rank: {torch.distributed.get_rank()} gather_object end!!!")
    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list = res_tup_list[:n_tasks]
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        # print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)
        # print(f"chores succeed", np.array(results).mean())

        eval_dir = os.path.join(eval_log_dir, flag)
        sr, sel, room = calculate_succ_rate_and_average_score_and_percentage_room(eval_dir)
        print(f"chores succeed {str(sr)}, sel {str(sel)}, room {str(room)}")

        sys.stdout.flush()  # 刷新缓冲区，确保输出即时被捕获
    return results


def evaluate_sequence(env, model, eval_loop_num, datasets, task, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False, flag='eval'):
    """
    Evaluates a sequence of language instructions.
    """
    success_counter = 0

    success = rollout(env, model, task, datasets, plans, debug, eval_log_dir, sequence_i, subtask_i=0, diverse_inst=diverse_inst, flag=flag)
    if success:
        success_counter += 1

    return success_counter

def rollout(env, model, task, datasets, plans, debug, eval_log_dir='', sequence_i=-1, subtask_i=0, diverse_inst=False, flag='eval'):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    def read_task(task):
        task_info = {}
        task_info["house_id"] = task["house_index"]
        task_info["sequence_i"] = task["sequence_i"]
        task_info["task_type"] = task["task_type"]
        task_info["base_position"] = task["agent_starting_position"]
        task_info["base_rotation"] = [0, task["agent_y_rotation"], 0]
        task_info["language"] = task["natural_language_spec"]
        task_info["expert_length"] = task["expert_length"]
        task_info["synsets"] = task["synsets"]
        task_info["broad_synset_to_object_ids"] = task["broad_synset_to_object_ids"]
        task_info["synset_to_object_ids"] = task["synset_to_object_ids"]
        task_info["followed_path"] = []

        return task_info

    def reset_env(env, scene, task):
        env.reset(scene=scene)
        # reset base pose
        agent_starting_position = task["base_position"] # array([ 2.25      , -0.90099216,  7.25      ])
        agent_starting_rotation = task["base_rotation"] # array([0., 0., 0.])
        env.step(
            action="Teleport",
            position=dict(x=agent_starting_position[0], y=agent_starting_position[1], z=agent_starting_position[2]),
            rotation=dict(x=agent_starting_rotation[0], y=agent_starting_rotation[1], z=agent_starting_rotation[2]),
            horizon=0,
            standing=True,
            renderImageSynthesis=True
        )
        # Do not display the unrealistic blue sphere on the agent's gripper
        env.step("ToggleMagnetVisibility", visible=False, raise_for_failure=True, renderImageSynthesis=True)
        # # reset gripper camera
        # gripper_pos = task["calib_rgb_gripper_position"][0] # array([ 2.30390525, -1.42482579,  7.19115162])
        # gripper_rot = task["calib_rgb_gripper_rotation"][0] # array([-30.00000381,  90.        ,   0.        ])
        # gripper_fov = task["cam_config_gripper_fov"][0] # 59.0
        # env.step(
        #     action="UpdateThirdPartyCamera",
        #     thirdPartyCameraId=0,
        #     position=dict(x=gripper_pos[0], y=-gripper_pos[1], z=gripper_pos[2]),
        #     rotation=dict(x=-gripper_rot[0], y=gripper_rot[1], z=gripper_rot[2]),
        #     fieldOfView=gripper_fov,
        #     renderImageSynthesis=True
        # )

    import math
    from spoc_robot_training.utils.type_utils import THORActions
    from spoc_robot_training.utils.constants.stretch_initialization_utils import (
        INTEL_VERTICAL_FOV,
        AGENT_RADIUS_LIST,
        AGENT_MOVEMENT_CONSTANT,
        ADDITIONAL_ARM_ARGS,
        AGENT_ROTATION_DEG,
        WRIST_ROTATION,
        ARM_MOVE_CONSTANT,
        HORIZON,
        ADDITIONAL_NAVIGATION_ARGS,
        STRETCH_COMMIT_ID,
        STRETCH_WRIST_BOUND_1,
        STRETCH_WRIST_BOUND_2,
    )
    from spoc_robot_training.utils.distance_calculation_utils import sum_dist_path, position_dist
    from spoc_robot_training.online_evaluation.online_evaluation_types_and_utils import calc_trajectory_room_visitation
    from spoc_robot_training.utils.data_generation_utils.navigation_utils import (
        get_rooms_polymap_and_type,
        # is_any_object_sufficiently_visible_and_in_center_frame,
        rotation_from,
        ALIGNMENT_THRESHOLD,
        PROP_VISIBLE_THRESHOLD,
    )
    from spoc_robot_training.environment.spoc_objects import SPOCObject
    from contextlib import contextmanager

    # from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
    def rotation_dist(a: Dict[str, float], b: Dict[str, float]):
        """Distance between rotations."""
        def deg_dist(d0: float, d1: float):
            dist = (d0 - d1) % 360
            return min(dist, 360 - dist)
        return sum(deg_dist(a[k], b[k]) for k in ["x", "y", "z"])

    def calc_arm_movement(arm_1, arm_2):
        total_dist = 0
        for k in ["x", "y", "z"]:
            total_dist += (arm_1[k] - arm_2[k]) ** 2

        return total_dist**0.5

    def get_current_agent_full_pose(controller):
        return {
            **controller.last_event.metadata["agent"],
            "arm": controller.last_event.metadata["arm"],
        }

    def get_relative_stretch_current_arm_state(controller):
        arm = controller.last_event.metadata["arm"]["joints"]
        z = arm[-1]["rootRelativePosition"]["z"]
        x = arm[-1]["rootRelativePosition"]["x"]
        assert abs(x - 0) < 1e-3
        y = arm[0]["rootRelativePosition"]["y"] - 0.16297650337219238
        return dict(x=x, y=y, z=z)
    
    def get_arm_wrist_rotation(controller):
        joint = controller.last_event.metadata["arm"]["joints"][-1]
        assert joint["name"] == "stretch_robot_wrist_2_jnt"
        return math.fmod(
            joint["rootRelativeRotation"]["w"] * joint["rootRelativeRotation"]["y"], 360
        )
    
    def step_env(controller, **kwargs):
        # if "renderImageSynthesis" not in kwargs:
        #     kwargs["renderImageSynthesis"] = should_render_image_synthesis
        kwargs["renderImageSynthesis"] = True
        
        return controller.step(**kwargs)

    def agent_step(controller, action):
        agents_full_pose_before_action = copy.deepcopy(
            dict(
                agent_pose=get_current_agent_full_pose(controller),
                arm_pose=get_relative_stretch_current_arm_state(controller),
                wrist=get_arm_wrist_rotation(controller),
            )
        )
        if action == THORActions.move_ahead:
            action_dict = dict(action="MoveAgent", ahead=AGENT_MOVEMENT_CONSTANT)
        elif action == THORActions.move_back:
            action_dict = dict(action="MoveAgent", ahead=-AGENT_MOVEMENT_CONSTANT)
        elif action in [
            THORActions.rotate_left,
            THORActions.rotate_right,
            THORActions.rotate_left_small,
            THORActions.rotate_right_small,
        ]:  #  add for smaller rotations
            if action == THORActions.rotate_right:
                degree = AGENT_ROTATION_DEG
            elif action == THORActions.rotate_left:
                degree = -AGENT_ROTATION_DEG
            elif action == THORActions.rotate_right_small:
                degree = AGENT_ROTATION_DEG / 5
            elif action == THORActions.rotate_left_small:
                degree = -AGENT_ROTATION_DEG / 5
            else:
                raise NotImplementedError

            action_dict = dict(action="RotateAgent", degrees=degree)
        elif action in [
            THORActions.move_arm_down,
            THORActions.move_arm_in,
            THORActions.move_arm_out,
            THORActions.move_arm_up,
            THORActions.move_arm_down_small,
            THORActions.move_arm_in_small,
            THORActions.move_arm_out_small,
            THORActions.move_arm_up_small,
        ]:
            base_position = get_relative_stretch_current_arm_state(controller)
            change_value = ARM_MOVE_CONSTANT
            small_change_value = ARM_MOVE_CONSTANT / 5
            if action == THORActions.move_arm_up:
                base_position["y"] += change_value
            elif action == THORActions.move_arm_down:
                base_position["y"] -= change_value
            elif action == THORActions.move_arm_out:
                base_position["z"] += change_value
            elif action == THORActions.move_arm_in:
                base_position["z"] -= change_value
            elif action == THORActions.move_arm_up_small:
                base_position["y"] += small_change_value
            elif action == THORActions.move_arm_down_small:
                base_position["y"] -= small_change_value
            elif action == THORActions.move_arm_out_small:
                base_position["z"] += small_change_value
            elif action == THORActions.move_arm_in_small:
                base_position["z"] -= small_change_value
            action_dict = dict(
                action="MoveArm",
                position=dict(x=base_position["x"], y=base_position["y"], z=base_position["z"]),
            )
        elif action in [
            THORActions.wrist_open,
            THORActions.wrist_close,
        ]:
            curr_wrist = get_arm_wrist_rotation(controller)
            if action == THORActions.wrist_open:
                rotation_value = -1 * min(
                    WRIST_ROTATION, abs(curr_wrist - (STRETCH_WRIST_BOUND_2 + 360))
                )
            else:  # action == THORActions.wrist_close:
                rotation_value = min(WRIST_ROTATION, abs(STRETCH_WRIST_BOUND_1 - curr_wrist))

            action_dict = dict(action="RotateWristRelative", yaw=rotation_value)
        elif action == THORActions.pickup:
            action_dict = dict(action="PickupObject")
        elif action == THORActions.dropoff:
            action_dict = dict(action="ReleaseObject")
        elif action in [
            THORActions.done,
            THORActions.sub_done,
        ]:
            action_dict = dict(action="Done")
        else:
            print("Action not defined")
            # pdb.set_trace()
            raise NotImplementedError("Action not defined")

        if action_dict["action"] in ["RotateWristRelative", "MoveArm"]:
            action_dict = {**action_dict, **ADDITIONAL_ARM_ARGS}
        elif action_dict["action"] == "MoveAgent":
            action_dict = {**action_dict, **ADDITIONAL_NAVIGATION_ARGS}

        event = step_env(controller, **action_dict)

        if action == THORActions.dropoff:
            step(controller, action="AdvancePhysicsStep", simSeconds=2)

        agents_full_pose_after_action = copy.deepcopy(
            dict(
                agent_pose=get_current_agent_full_pose(controller),
                arm_pose=get_relative_stretch_current_arm_state(controller),
                wrist=get_arm_wrist_rotation(controller),
            )
        )

        # test for checking move arm is failing or not
        #  return false if arm move  is called but pose is not changed
        if action in THORActions.ARM_ACTIONS:
            event.metadata["lastActionSuccess"] = (
                calc_arm_movement(
                    agents_full_pose_before_action["arm_pose"],
                    agents_full_pose_after_action["arm_pose"],
                )
                > 1e-3
            )

        if action in [
            THORActions.wrist_open,
            THORActions.wrist_close,
        ]:
            event.metadata["lastActionSuccess"] = (
                abs(
                    agents_full_pose_before_action["wrist"] - agents_full_pose_after_action["wrist"]
                )
                > 1e-3
            )

        # Only a failure moving if we don't move enough
        if action in THORActions.MOVE_ACTIONS:
            event.metadata["lastActionSuccess"] = (
                position_dist(
                    agents_full_pose_before_action["agent_pose"]["position"],
                    agents_full_pose_after_action["agent_pose"]["position"],
                )
                > 1e-2
            )

        # Only a failure rotating if we don't rotate enough
        if action in THORActions.ROTATE_ACTIONS:
            event.metadata["lastActionSuccess"] = (
                rotation_dist(
                    agents_full_pose_before_action["agent_pose"]["rotation"],
                    agents_full_pose_after_action["agent_pose"]["rotation"],
                )
                > 2
            )
        return event
    
    @contextmanager
    def include_object_metadata_context(controller):
        needs_reset = len(controller.last_event.metadata["objects"]) == 0
        try:
            if needs_reset:
                controller.step("ResetObjectFilter")
                assert controller.last_event.metadata["lastActionSuccess"]
            yield None
        finally:
            if needs_reset:
                obj_meta = controller.last_event.metadata["objects"]
                controller.step("SetObjectFilter", objectIds=[])
                controller.last_event.metadata["objects"] = obj_meta
                assert controller.last_event.metadata["lastActionSuccess"]

    def get_object(controller, object_id: str, include_receptacle_info: bool = False):
        """
        NOTE: It may be much less efficient to `include_receptacle_info` than to not.

        :param object_id:
        :param include_receptacle_info:
        :return:
        """
        if include_receptacle_info or any(
            object_id == o["objectId"] for o in controller.last_event.metadata["objects"]
        ):
            with include_object_metadata_context(controller):
                return SPOCObject(controller.last_event.get_object(object_id))

        meta = controller.step(
            action="GetObjectMetadata", objectIds=[object_id], raise_for_failure=True
        ).metadata["actionReturn"][0]

        del meta[
            "parentReceptacles"
        ]  # This will always be None when using GetObjectMetadata so remove it so there is no ambiguity
        return SPOCObject(meta)

    def get_object_position(controller, object_id):
        try:
            return get_object(controller, object_id)["position"]
        except KeyError:
            event = get_object(controller, object_id)
            print(event)
            print(object_id)

    def get_agent_alignment_to_object(controller, object_id: str, use_arm_orientation: bool = False):
        current_agent_pose = get_current_agent_full_pose(controller) # 本体位置和朝向
        if use_arm_orientation: # 如果是夹爪相机看到的，那么绕y轴旋转90度到夹爪相机朝向
            current_agent_pose = copy.deepcopy(current_agent_pose)
            current_agent_pose["rotation"]["y"] += 90
        object_location = get_object_position(controller, object_id) # 获取物体的位置
        vector = rotation_from(current_agent_pose, object_location) # 表示物体相对于机器人的旋转角。如果是-10度，那么说明机器人要顺时针方向旋转10度才和物体同向
        return vector

    def manipulation_camera_segmentation(
        controller,
    ):
        if controller.last_event.instance_segmentation_frame is None:
            controller.step("Pass", renderImageSynthesis=True)
            assert controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a manipulation_camera_segmentation"
            )

        return controller.last_event.third_party_instance_masks[0]

    def navigation_camera_segmentation(
        controller,
    ):
        if controller.last_event.instance_segmentation_frame is None:
            controller.step("Pass", renderImageSynthesis=True)
            assert controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a navigation_camera_segmentation"
            )

        return controller.last_event.instance_masks

    def is_any_object_sufficiently_visible_and_in_center_frame(
        controller: "StretchController",
        object_ids: List[str],
        scale: float = 1.5e4,
        manipulation_camera=False,
        absolute_min_pixels: int = 200, # 表示物体可见的最小像素数量（默认为 200 个像素）
    ):
        # Note: this code assumes INTEL_VERTICAL_FOV = 59 is used
        # Note: small object of e.g. 0.15 * 0.15 -> 0.0225 m2 (scale ~10,000 for 200 pixels) 如果物体的表面积为0.0225 m2，那么经过scale（10,000）放大为200个像素，相当于认为1cm2等于1像素
        # Also, constraining the maximum threshold to be 1000 (the old one for non-SMALL_TYPES), 表示物体可见的最大像素数量（默认为 1000 个像素）
        # and the minimum to be 200 (the old one for SMALL_TYPES). 表示物体可见的最小像素数量（默认为 200 个像素）
        # Note: Using default scale=15,000 instead of 10,000 as `ProportionOfObjectVisible` resolves it. 经过scale（15,000）放大为200个像素，相当于认为1cm2等于1.5像素
        # Note: default image height = 224 -> 224 ^ 2 = 50176 pixels is the area for a 1:1 image shape ratio. 将图像尺寸调整为224*224的正方形，即视野中央位置
        # 计算每个物体的像素上限值
        scale_to_apply = scale * (224 ** 2) / 50176 # 物体表面积与像素个数的比例scale为15,000
        pixel_mass_thresholds = {} # 存放物体的像素上限值
        for oid in object_ids:
            if manipulation_camera: # 如果是夹爪相机，直接认为物体的像素上限值为200
                pixel_mass_thresholds[oid] = 200
            else: # 如果是本体相机，需要通过物体表面积来计算像素上限值
                # # d1, d2, d3 are the dimensions of the bbox, in meters
                try:
                    _, mags = get_basis_for_3d_box(controller.get_object(oid)) # 获取物体的 3D 包围盒尺寸（d1, d2, d3），单位为m
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    # This is a hack to deal with the fact that the basis `get_basis_for_3d_box` may
                    # fail for some objects. In that case, we just use the default value of 200.
                    pixel_mass_thresholds[oid] = 200 # 对于无法获取 3D 包围盒的物体，直接认为物体的像素上限值为200
                    continue
                d1, d2, d3 = tuple(mags)
                largest_bbox_face_area = max(d1 * d2, d2 * d3, d3 * d1) # 根据物体的 3D 包围盒尺寸（d1, d2, d3），以物体6个面的最大面积作为物体的像素上限值
                pixel_mass_thresholds[oid] = max(
                    min(scale_to_apply * largest_bbox_face_area, 1000), absolute_min_pixels # 物体的像素上限值被限制在 200 到 1000 之间
                )
        # 计算每个物体的对齐度和像素占比
        oid_to_seg = None
        target_visibility_quant = {} # 存放物体的像素个数对齐度和像素占比
        for oid in object_ids:
            alignment = abs(
                get_agent_alignment_to_object(controller, oid, use_arm_orientation=manipulation_camera) # 表示物体相对于机器人的旋转角。如果是-10度，那么说明机器人要顺时针方向旋转10度才和物体同向
            )
            if alignment <= ALIGNMENT_THRESHOLD: # 只有物体相对于机器人的旋转角小于+-10度，物体才被认为在中央视野内
                if oid_to_seg is None: # 获取instance_masks
                    # Have to do this here to avoid loading the segmentation masks
                    # when we don't need to
                    oid_to_seg = (
                        manipulation_camera_segmentation(controller)
                        if manipulation_camera
                        else navigation_camera_segmentation(controller)
                    )
                try:
                    pixel_mass = oid_to_seg[oid].sum() # 获取物体在图像中的像素数量
                    empty_top = (
                        None  # MANIP CAMERA ONLY: Make sure there are at least 10% empty pixels on top of the image
                        if not manipulation_camera
                        else (
                            oid_to_seg[oid][: int(0.1 * 224)] == 0 # 检查图像顶部 10% 的区域是否为空，确保物体没有占据图像顶部
                        ).all()
                    )
                except KeyError:
                    pixel_mass = 0
                    empty_top = False

                target_visibility_quant[oid] = {
                    "alignment": alignment, # 对齐度
                    "pixel_mass": pixel_mass, # 像素占比
                    "empty_top": empty_top, # 顶部是否为空
                }

        sufficient_visibility = False
        for obj_id, obj_data in target_visibility_quant.items():
            if obj_data["alignment"] >= ALIGNMENT_THRESHOLD: # 对齐度需要在+-10度以内，否则pass
                continue

            if obj_data["pixel_mass"] < absolute_min_pixels: # 像素占比需要超过200个像素，否则pass
                continue

            if obj_data["pixel_mass"] <= pixel_mass_thresholds[obj_id]: # 检查物体的像素质量是否大于像素上限值，也就是自己的最大的一个面
                # The pixel mass threshold can be overly strict so we include here an approximate check 这样的检查过于严格，所以考虑可见比例
                # to see if the object is > prop_visible_threshold visible and, if it is, we say its ok.
                prop_visible = controller.step(
                    action="ProportionOfObjectVisible",
                    objectId=list(target_visibility_quant.keys())[0],
                ).metadata["actionReturn"]
                if prop_visible < PROP_VISIBLE_THRESHOLD: # 物体的可见比例需要超过0.8，否则pass
                    continue

            if obj_data["empty_top"] is not None and (not obj_data["empty_top"]):
                continue

            sufficient_visibility = True
            break # 只要目标物体中有一个可见，就返回

        return sufficient_visibility

    def get_top_down_path_view(
        controller,
        agent_path,
        targets_to_highlight=None,
        orthographic=True,
        map_height_width=(1000, 1000),
        path_width=0.045,
    ):
        thor_controller = controller.controller if hasattr(controller, "controller") else controller

        original_hw = thor_controller.last_event.frame.shape[:2]

        if original_hw != map_height_width:
            event = thor_controller.step(
                "ChangeResolution", x=map_height_width[1], y=map_height_width[0], raise_for_failure=True
            )

        if len(thor_controller.last_event.third_party_camera_frames) < 2:
            event = thor_controller.step("GetMapViewCameraProperties", raise_for_failure=True)
            cam = copy.deepcopy(event.metadata["actionReturn"])
            if not orthographic:
                bounds = event.metadata["sceneBounds"]["size"]
                max_bound = max(bounds["x"], bounds["z"])

                cam["fieldOfView"] = 50
                cam["position"]["y"] += 1.1 * max_bound
                cam["orthographic"] = False
                cam["farClippingPlane"] = 50
                del cam["orthographicSize"]

            event = thor_controller.step(
                action="AddThirdPartyCamera",
                **cam,
                skyboxColor="white",
                raise_for_failure=True,
            )

        waypoints = []
        for target in targets_to_highlight or []:
            target_position = get_object_position(thor_controller, target)
            target_dict = {
                "position": target_position,
                "color": {"r": 1, "g": 0, "b": 0, "a": 1},
                "radius": 0.5,
                "text": "",
            }
            waypoints.append(target_dict)

        if len(agent_path) != 0:
            thor_controller.step(
                action="VisualizeWaypoints",
                waypoints=waypoints,
                raise_for_failure=True,
            )
            # put this over the waypoints just in case
            event = thor_controller.step(
                action="VisualizePath",
                positions=agent_path,
                pathWidth=path_width,
                raise_for_failure=True,
            )
            thor_controller.step({"action": "HideVisualizedPath"})

        map = event.third_party_camera_frames[-1]

        if original_hw != map_height_width:
            thor_controller.step(
                "ChangeResolution", x=original_hw[1], y=original_hw[0], raise_for_failure=True
            )

        return map

    # 模型reset
    model.reset()
    # env reset
    json_data = read_task(task)
    house_id = json_data["house_id"] # 68
    scene = datasets[int(house_id)]
    reset_env(env, scene, json_data)
    
    # get lang annotation for subtask
    lang_annotation = json_data["language"] # 'go to an apple and take that apple'

    debug = True
    done = False # zyf
    if debug:
        img_queue = []
        img_queue2 = []
    planned_actions = []
    action_list = []
    trajectory = []
    step_len = EP_LEN

    room_poly_map, room_type_dict = get_rooms_polymap_and_type(scene)

    def calculate_sel(done, task_info, eps_len):
        if not done:
            sel = 0.0
        else:
            S_i = 1.0
            W_i = task_info["expert_length"]
            E_i = eps_len

            sel = S_i * W_i / max(W_i, E_i)

        return f"{sel:.4f}"
    
    def calculate_room_visited(room_poly_map, trajectory):
        percentage_visited, total_visited = calc_trajectory_room_visitation(
            room_poly_map, trajectory
        )

        return f"{percentage_visited:.4f}", f"{total_visited:.4f}"

    for step in range(step_len):
        if model.replan != -1 and step % model.replan == 0:
            if model.model.module.refresh != -1:
                model.model.module.lang_encoder.lm_head.hidden_state = None
                model.model.module.lang_encoder.lm_head.history_memory = model.model.module.lang_encoder.lm_head.history_memory[-model.refresh:]
            else:
                model.reset()
        obs = {
            "rgb_obs":{
                "rgb_static":  env.last_event.frame, 
                "rgb_gripper": env.last_event.third_party_camera_frames[0][:, :, :3] 
            },
            "robot_obs": np.zeros(20),
        }

        base_pos = env.last_event.metadata["agent"]["position"]
        json_data["followed_path"].append(base_pos)
        base_pos = np.array(list(base_pos.values()))
        trajectory.append(base_pos)

        action, obs_preds_rgb_img, obs_preds_gripper_img, obs_pred = model.step(obs, lang_annotation, env, (len(planned_actions) == 0), step)

        if obs_pred is not None:
            voxel_range = [[-0.5*4, 0.5*4], [-1.0, 3.0], [0.0, 1.0]]
            voxel_size = [0.025*4, 0.025*4, 0.025*2]
            vfe_generator = OccupancyVFE(voxel_range, voxel_size)
            obs_pred[0][:,:,:,0] = torch.sigmoid(obs_pred[0][:,:,:,0])
            obs_pred = np.array(obs_pred[0].cpu())
            point,rgb = vfe_generator.decode_occupied_grid(obs_pred)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            pcd_dir = os.path.join(eval_log_dir, f'pcd/{sequence_i}-{subtask_i}-{lang_annotation}')
            if not os.path.exists(pcd_dir):
                try: os.makedirs(pcd_dir)
                except: print('dirs_maked')
            pcd_path = os.path.join(pcd_dir, f'pcd_{step}.pcd')
            o3d.io.write_point_cloud(pcd_path, pcd)

        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        if model.use_diff:
            model.action_hist_queue.append(action)

        if debug:
            # obs_preds_rgb_img, obs_preds_gripper_img
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


        action_idx = int(action[-1]) # 只需要最后一维
        from spoc_robot_training.utils.constants.stretch_initialization_utils import ALL_STRETCH_ACTIONS
        action_str = ALL_STRETCH_ACTIONS[action_idx]
        event = agent_step(env, action_str)

        def is_successful_fetch(env, task_info):
            object_type = task_info["synsets"][0]
            target_held_objects = [
                x
                for x in env.last_event.metadata["arm"]["heldObjects"]
                if x in task_info["broad_synset_to_object_ids"][object_type]
            ]
            return len(target_held_objects) > 0  

        def get_visible_objects(
            env,
            which_camera: Literal["nav", "manip", "both"] = "nav",
            maximum_distance=2,
            filter_object_ids: Optional[Sequence[str]] = None,
        ):
            assert which_camera in ["nav", "manip", "both"]

            if filter_object_ids is not None and not isinstance(filter_object_ids, List):
                filter_object_ids = list(filter_object_ids)

            visible_objects = set()
            if which_camera in ["nav", "both"]:
                visible_objects.update(
                    env.step(
                        "GetVisibleObjects",
                        maxDistance=maximum_distance,
                        objectIds=filter_object_ids,
                    ).metadata["actionReturn"]
                )

            if which_camera in ["manip", "both"]:
                visible_objects.update(
                    env.step(
                        "GetVisibleObjects",
                        maxDistance=maximum_distance,
                        thirdPartyCameraIndex=0,
                        objectIds=filter_object_ids,
                    ).metadata["actionReturn"]
                )

            return list(visible_objects)      

        def object_is_visible_in_camera(
            env, object_id, which_camera: Literal["nav", "manip", "both"] = "nav", maximum_distance=2
        ):
            return (
                len(
                    get_visible_objects(
                        env,
                        which_camera=which_camera,
                        maximum_distance=maximum_distance,
                        filter_object_ids=[object_id],
                    )
                )
                != 0
            )

        def is_successful_nav(env, task_info, strict_success=True):
            object_type = task_info["synsets"][0]
            visible_targets = [
                oid
                for oid in task_info["broad_synset_to_object_ids"][object_type]
                if object_is_visible_in_camera(
                    env, oid, which_camera="nav", maximum_distance=2
                )
            ]
            if not strict_success: # 如果不严格的话，看到就行了，不需要在视野重要
                return len(visible_targets) > 0
            elif len(visible_targets) == 0: # 如果严格的话，没看到直接返回，看到的需要判断是否在视野中央
                return False

            return is_any_object_sufficiently_visible_and_in_center_frame(
                controller=env, object_ids=visible_targets
            )

        def is_successful(env, task_info):
            # task_info["task_type"]同spoc-robot-training/training/offline/dataset_mixtures.py
            if task_info["task_type"] in ["FetchType", "PickupType"]:
                return is_successful_fetch(env, task_info)
            elif task_info["task_type"] in ["ObjectNavRoom", "ObjectNavType", "ObjectNavRelAttribute", "ObjectNavAffordance", "ObjectNavLocalRef", "ObjectNavDescription"]: 
                return is_successful_nav(env, task_info)
            elif task_info["task_type"] in ["RoomNav"]:
                raise NotImplementedError
            elif task_info["task_type"] in ["RoomVisit"]:
                raise NotImplementedError

        is_strict = True

        if not is_strict:
            if is_successful(env, json_data):
                done = True 
        else:
            if action_str == "end" and is_successful(env, json_data): # 太严格了
                done = True 

        if done:  
            print("house_id:", json_data["house_id"])
            if debug:
                target_ids = None
                if "synset_to_object_ids" in json_data:
                    target_ids = list(
                        chain.from_iterable(json_data["synset_to_object_ids"].values())
                    )

                top_down_frame = get_top_down_path_view(env, json_data["followed_path"], target_ids)
                top_down_frame = np.ascontiguousarray(top_down_frame)
                print(colored("success", "green"), end=" ")
                eval_dir = os.path.join(eval_log_dir, flag)
                sel = calculate_sel(done, json_data, step + 1) # step从0开始计算，实际步数需要+1
                percentage_visited, total_visited = calculate_room_visited(room_poly_map, trajectory)
                if not os.path.exists(eval_dir):
                    try:
                        os.makedirs(eval_dir)
                    except:
                        print('dirs_maked')
                img_clip = ImageSequenceClip(img_queue, fps=30)
                img_clip.write_gif(os.path.join(eval_dir, f'{json_data["sequence_i"]}_{subtask_i}_{lang_annotation.replace(" ", "-")}_succ_{sel}_{percentage_visited}.gif'), fps=30) # , logger=None
                plt.imsave(fname=os.path.join(eval_dir, f'{json_data["sequence_i"]}_{subtask_i}_{lang_annotation.replace(" ", "-")}_succ_{sel}_{percentage_visited}.gif_topdown.png'), arr=top_down_frame, dpi=500)
            return True
    if debug:
        target_ids = None
        if "synset_to_object_ids" in json_data:
            target_ids = list(
                chain.from_iterable(json_data["synset_to_object_ids"].values())
            )

        top_down_frame = get_top_down_path_view(env, json_data["followed_path"], target_ids)
        top_down_frame = np.ascontiguousarray(top_down_frame)
        print(colored("fail", "red"), end=" ")
        # accuracy = sum(action_list) / len(action_list)
        # print(f"gt一致率: {accuracy:.2f}")
        eval_dir = os.path.join(eval_log_dir, flag)
        sel = calculate_sel(done, json_data, step_len)
        percentage_visited, total_visited = calculate_room_visited(room_poly_map, trajectory)
        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except:
                print('dirs_maked')
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_dir, f'{json_data["sequence_i"]}_{subtask_i}_{lang_annotation.replace(" ", "-")}_fail_{sel}_{percentage_visited}.gif'), fps=30) # , logger=None
        plt.imsave(fname=os.path.join(eval_dir, f'{json_data["sequence_i"]}_{subtask_i}_{lang_annotation.replace(" ", "-")}_fail_{sel}_{percentage_visited}.gif_topdown.png'), arr=top_down_frame, dpi=500)
    return False

def eval_one_epoch_calvin(args, model, dataset_path, image_processor, tokenizer, future_act_len=-1):

    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    wrapped_model = ModelWrapper(args, model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=args.n_obs_steps, future_act_len=future_act_len)
    evaluate_policy(wrapped_model, env, 0, args.calvin_conf_path)


def eval_one_epoch_calvin_ddp(args, model=None, dataset_path=None, image_processor=None, tokenizer=None, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False):

    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type=="diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    wrapped_model = ModelWrapper(args, model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=hist_len, future_act_len=future_act_len)
    evaluate_policy_ddp(wrapped_model, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst, args=args, only_single_task=args.only_single_task, resume=args.resume)

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