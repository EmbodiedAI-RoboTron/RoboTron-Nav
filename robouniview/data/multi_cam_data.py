import ast
import argparse
from cgitb import text
import functools
import io, time
import json
import logging
import math
import os
import yaml
import random
import sys
import glob
import tarfile
from dataclasses import dataclass
from multiprocessing import Value
import zipfile
import cv2, copy
import braceexpand
import torch
import torchvision
import webdataset as wds
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
# import pybullet as pb
sys.path.append('~/yanfeng/project/robotic/calvin/calvin_models')
sys.path.append('~/yanfeng/project/robotic/calvin/calvin_env')
sys.path.append('~/yanfeng/project/robotic/calvin/calvin_env/tacto_env')
from calvin_agent.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_state,
)
sys.path.append('~/liufanfan/workspace/RoboUniView/open_flamingo')
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
# 仿真器
from scipy.spatial.transform import Rotation as scipyR 
import hydra , pybullet
import open3d as o3d
from omegaconf import OmegaConf
from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    print_and_save,
)
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset
from robouniview.data.real_dataset_hdf5 import RealDatasetHDF5
import re,h5py
import torchvision.transforms as transforms
from robouniview.data.data_utils import OccupancyVFE
# from .zipreader import ZipReader
from concurrent.futures import ThreadPoolExecutor

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from pathlib import Path
from typing import Dict, Tuple, Union
from robouniview.data.vl_dataset import CaptionDataset, VQADataset
hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)
import random
import torchvision.transforms.functional as visionF
from typing import Any, Dict, List, Tuple, Callable
from itertools import chain
from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern
import pickle
import torch.nn as nn
import torch.nn.functional as F
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug


def load_from_zip(zip_ref, filename):
    with zip_ref.open(filename) as file:
        return io.BytesIO(file.read())


def get_validation_window_size(
    idx: int, min_window_size: int, max_window_size: int
) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }      
)

prop_state = DictConfig(
    {
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)


keys_calib = ['static_extrinsic_matrix','static_intrinsic_matrix','static_distCoeffs_matrix',
                'gripper_extrinsic_matrix','gripper_intrinsic_matrix','gripper_distCoeffs_matrix','state_matrix','static_fov', 'gripper_fov'] 


class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        # print('ws {}, min_ws {}, max_ws {}'.format(self.window_size, self.max_window_size, self.min_window_size))
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
        with open('~/yanfeng/project/robotic/RoboFlamingo/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        # assert (
        #     "validation" in self.abs_datasets_dir.as_posix()
        #     or "training" in self.abs_datasets_dir.as_posix()
        # )
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
       
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            # if len(rgb_obs.shape) != 4:
            #     rgb_obs = np.expand_dims(rgb_obs, axis=0)
            # assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                # seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
                seq_rgb_obs_ = rgb_obs
            else:  # episode loader 
                # seq_rgb_obs_ = torch.from_numpy(
                #     rgb_obs[seq_idx : seq_idx + window_size]
                # ).byte()
                seq_rgb_obs_ = rgb_obs[seq_idx : seq_idx + window_size]
            
            if rgb_obs_key in transforms:
                # seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
                # seq_rgb_obs_ = torch.stack([transforms[rgb_obs_key](Image.fromarray(img)) for img in seq_rgb_obs_])
                seq_rgb_obs_ = torch.stack([transforms[rgb_obs_key](img) for img in seq_rgb_obs_])
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_ # 13 3 224 224
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}
    
    def process_calib(
        self,
        episode: Dict[str, np.ndarray],
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        # keys_calib = ['static_extrinsic_matrix','static_intrinsic_matrix','static_distCoeffs_matrix',
        #                 'gripper_extrinsic_matrix','gripper_intrinsic_matrix','gripper_distCoeffs_matrix','state_matrix'] 
        seq_calib_obs_dict = {}
        for _, calib_obs_key in enumerate(keys_calib):
            calib_obs = episode[calib_obs_key]
            if window_size == 0 and seq_idx == 0:  # single file loader
                seq_calib_obs_ = torch.from_numpy(calib_obs)
            else:
                seq_calib_obs_ = torch.from_numpy(calib_obs[seq_idx : seq_idx + window_size])
            seq_calib_obs_dict[calib_obs_key] = seq_calib_obs_
        
        return {"calib_obs": seq_calib_obs_dict}

    def process_his(
        self,
        episode: Dict[str, np.ndarray],
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        keys_his = ['his_vision','his_pose']

        seq_his_obs_dict = {}
        for _, his_obs_key in enumerate(keys_his):
            his_obs = episode[his_obs_key]
            if window_size == 0 and seq_idx == 0:  # single file loader
                seq_his_obs_ = torch.from_numpy(his_obs)
            else:
                seq_his_obs_ = torch.from_numpy(his_obs[seq_idx : seq_idx + window_size])
            seq_his_obs_dict[his_obs_key] = seq_his_obs_
        
        return {"his_obs": seq_his_obs_dict}
    
    def process_pcd(
        self,
        episode: Dict[str, np.ndarray],
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        keys_pcd = ['pcd'] 
        seq_pcd_obs_dict = {}
        for _, pcd_obs_key in enumerate(keys_pcd):
            pcd_obs = episode[pcd_obs_key] # 13 80 80 40 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                seq_pcd_obs_ = torch.from_numpy(pcd_obs)
            else:
                seq_pcd_obs_ = torch.from_numpy(pcd_obs[seq_idx : seq_idx + window_size])
            seq_pcd_obs_dict[pcd_obs_key] = seq_pcd_obs_
        
        return {"pcd_obs": seq_pcd_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        resample = True
        max_data_fetch_iteration = 300

        for fetch_iteration in range(max_data_fetch_iteration):
            try:
            # if 1:
                if not isinstance(idx, Tuple):
                    # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
                    # acts like Constant dataset. Currently, used for language data
                    if self.min_window_size == self.max_window_size:
                        window_size = self.max_window_size
                    elif self.min_window_size < self.max_window_size:
                        window_size = self._get_window_size(idx)
                    else:
                        logger.error(
                            f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                        )
                        raise ValueError
                else:
                    idx, window_size = idx
        
                head = False
                sequence = self._get_sequences(idx, window_size, head=head)
                if sequence == 0:
                    idx = random.randint(0, len(self) - 1)
                    logger.info(
                            f"env_resample"
                        )
                else:
                    if self.pad:
                        pad_size = self._get_pad_size(sequence)
                        sequence = self._pad_sequence(sequence, pad_size, head=head)
                    
                    # import copy
                    # new_list = []
                    # np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
                    # for i in range(np_rgb.shape[0]):
                    #     new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
                    # sequence["rgb_obs"]["rgb_static"] = new_list
                    # new_list = []
                    # np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
                    # for i in range(np_gripper.shape[0]):
                    #     new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8))) #uint8
                    # sequence["rgb_obs"]["rgb_gripper"] = new_list
                    return sequence
            
            except Exception as e:
                print(f"Resample warning:dataset warning{fetch_iteration}, {e}")
                #pass
            if resample:
                idx = random.randint(0, len(self) - 1)
            else:
                return None

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        episode = self._load_episode(idx, window_size)
        if episode == 0:
            return 0
        
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_calib_obs = self.process_calib(episode)
        seq_his_obs = self.process_his(episode)
        seq_pcd_obs = self.process_pcd(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **seq_calib_obs,
            **seq_pcd_obs,
            **seq_his_obs,            
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int: #
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int: #yiyang_question
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)}) # YF：原来不传入head参数，为啥robot_obs不使用head呢？
        seq.update(
            {
                "calib_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["calib_obs"].items()
                }
            }
        )
        seq.update(
            {
                "pcd_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["pcd_obs"].items()
                }
            }
        )
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head), # XYZ补0，爪子不动
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.
        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info


class DebugDataset(Dataset):
    def __init__(self, **kwargs: Any,):
        super().__init__()
    def __len__(self) -> int:
        return 10000
    def __getitem__(self, index):
        window_size = 8
        rgb = torch.randn(window_size, 3, 200, 200)
        gripper = torch.randn(window_size, 84, 84)
        state = torch.randn(window_size, 15)


class BaseMultiDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        colour_aug=[0,0,0,0],
        data_path_list = [],
        state_matrixs_path = '',
        data_tasks_groups = None,
        env_resample = False,
        only_single_task=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.env_resample = env_resample
        self.data_tasks_groups = data_tasks_groups
        self.state_matrixs_path = state_matrixs_path
        self.save_format = save_format
        self.data_path_list = data_path_list
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.colour_aug = colour_aug
        if sum(colour_aug) ==0:
            self.use_colour_aug = False
        else:
            self.use_colour_aug = True
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.only_single_task = only_single_task

        self.ColorJitter = ColorJitter_ctm(colour_aug[0], colour_aug[1], colour_aug[2], colour_aug[3])
        voxel_range = [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]
        voxel_size = [0.0125, 0.0125, 0.0125*2]
        self.vfe_generator = OccupancyVFE(voxel_range, voxel_size)
    
        self.tmp=[] 
        self.data_name = None

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = False
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

    def _prepare_eposides(self, episodes):
        
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        keys.remove("rgb_static")
        keys.remove("rgb_gripper") # keys = ['robot_obs', 'rel_actions', 'scene_obs']
        # keys_calib = ['static_extrinsic_matrix','static_intrinsic_matrix','static_distCoeffs_matrix',
        #               'gripper_extrinsic_matrix','gripper_intrinsic_matrix','gripper_distCoeffs_matrix','state_matrix','static_fov', 'gripper_fov'] 
        keys_rgb = ['rgb_static','rgb_gripper']
        keys_his = ['his_vision','his_pose']

        calibs = []
        pcds = []
        rgbs = []
        colour_aug_random = random.randint(0, 10)
        for i, ep in enumerate(episodes):
            rgb = {}
            calib = ep['calib']
            cam_config = ep['cam_config']
            static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
            gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']
            static_cam = cam(static_extrinsic_matrix, cam_config['static']['height'], cam_config['static']['width'], cam_config['static']['fov'])
            gripper_cam = cam(gripper_extrinsic_matrix, cam_config['gripper']['height'], cam_config['gripper']['width'], cam_config['gripper']['fov'])
            static_pcd = deproject(static_cam, ep['depth_static'], homogeneous=False, sanity_check=False).transpose(1, 0)
            gripper_pcd = deproject(gripper_cam, ep['depth_gripper'], homogeneous=False, sanity_check=False).transpose(1, 0)
            cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)
            rgb['rgb_static'] = ep['rgb_static'] # Image.fromarray(ep['rgb_static'])
            rgb['rgb_gripper'] = ep['rgb_gripper'] # Image.fromarray(ep['rgb_gripper'])
            if colour_aug_random>2 and self.use_colour_aug:
                if i == 0: rgb['rgb_static'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.ColorJitter(rgb['rgb_static'])
                else: rgb['rgb_static'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.ColorJitter(rgb['rgb_static'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                rgb['rgb_gripper'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.ColorJitter(rgb['rgb_gripper'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
            # rgb['rgb_static'] = np.array(rgb['rgb_static'])
            # rgb['rgb_gripper'] = np.array(rgb['rgb_gripper'])
            static_rgb = np.reshape(np.array(rgb['rgb_static']), (-1, 3))
            gripper_rgb =  np.reshape(np.array(rgb['rgb_gripper']), (-1, 3))
            pcd_rgb = np.concatenate([static_rgb, gripper_rgb], axis=0)
            pcd_rgb = pcd_rgb/255
            pcd = self.vfe_generator.generate(cloud[:, :3], pcd_rgb)
            if 0:
                _pcd = o3d.geometry.PointCloud()
                point_tmp, rgb_tmp = self.vfe_generator.decode_occupied_grid_with_range(pcd)
                _pcd.points = o3d.utility.Vector3dVector(point_tmp)
                _pcd.colors = o3d.utility.Vector3dVector(rgb_tmp)
                o3d.io.write_point_cloud("tmp.pcd", _pcd)
                _pcd.points = o3d.utility.Vector3dVector(static_pcd)
                _pcd.colors = o3d.utility.Vector3dVector(static_rgb/255)
                o3d.io.write_point_cloud("tmp1.pcd", _pcd)
                _pcd.points = o3d.utility.Vector3dVector(gripper_pcd)
                _pcd.colors = o3d.utility.Vector3dVector(gripper_rgb/255)
                o3d.io.write_point_cloud("tmp2.pcd", _pcd)
            calib['static_extrinsic_matrix'] = static_extrinsic_matrix * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
            calib['static_intrinsic_matrix'] = calib['rgb_static']['intrinsic_matrix']
            calib['static_distCoeffs_matrix'] = calib['rgb_static']['distCoeffs_matrix']
            calib['gripper_extrinsic_matrix'] = gripper_extrinsic_matrix * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
            calib['gripper_intrinsic_matrix'] = calib['rgb_gripper']['intrinsic_matrix']
            calib['gripper_distCoeffs_matrix'] = calib['rgb_gripper']['distCoeffs_matrix']
            calib['state_matrix'] = ep['state_matrix'] * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
            calib['static_fov'] = np.array([cam_config['static']['fov'], cam_config['static']['height'], cam_config['static']['width']])
            calib['gripper_fov'] = np.array([cam_config['gripper']['fov'], cam_config['gripper']['height'], cam_config['gripper']['width']])
            rgbs.append(rgb)
            pcds.append(pcd)
            calibs.append(calib)
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode.update({key: np.stack([calib[key] for calib in calibs]) for key in keys_calib})
        episode.update({key: [rgb[key] for rgb in rgbs] for key in keys_rgb}) # episode.update({key: np.stack([rgb[key] for rgb in rgbs]) for key in keys_rgb})
        episode['pcd'] = np.stack(pcds)
        episode.update({key: np.stack([ep[key] for ep in episodes]) for key in keys_his})

        return episode

    # def _prepare_eposides(self, episodes):
        
    #     keys = list(chain(*self.observation_space.values()))
    #     keys.remove("language")
    #     keys.append("scene_obs")
    #     keys.remove("rgb_static")
    #     keys.remove("rgb_gripper")
    #     # keys_calib = ['static_extrinsic_matrix','static_intrinsic_matrix','static_distCoeffs_matrix',
    #     #               'gripper_extrinsic_matrix','gripper_intrinsic_matrix','gripper_distCoeffs_matrix','state_matrix','static_fov', 'gripper_fov'] 
    #     keys_rgb = ['rgb_static','rgb_gripper']

    #     calibs = []
    #     pcds = []
    #     rgbs = []
    #     ColorJitter_func = None
    #     colour_aug_random = random.randint(0, 10)
    #     # i=0
    #     if 1:
    #         ep = episodes[0]
    #         rgb = {}
    #         calib = ep['calib']
    #         cam_config = ep['cam_config']
    #         static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
    #         gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']
    #         static_cam = cam(static_extrinsic_matrix, cam_config['static']['height'], cam_config['static']['width'], cam_config['static']['fov'])
    #         gripper_cam = cam(gripper_extrinsic_matrix, cam_config['gripper']['height'], cam_config['gripper']['width'], cam_config['gripper']['fov'])
    #         static_pcd = deproject(static_cam, ep['depth_static'], homogeneous=False, sanity_check=False).transpose(1, 0)
    #         gripper_pcd = deproject(gripper_cam, ep['depth_gripper'], homogeneous=False, sanity_check=False).transpose(1, 0)
    #         cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)
    #         rgb['rgb_static'] = ep['rgb_static'] # Image.fromarray(ep['rgb_static'])
    #         rgb['rgb_gripper'] = ep['rgb_gripper'] # Image.fromarray(ep['rgb_gripper'])
    #         if colour_aug_random>2 and self.use_colour_aug:
    #             rgb['rgb_static'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.ColorJitter(rgb['rgb_static'])
    #             rgb['rgb_gripper'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.ColorJitter(rgb['rgb_gripper'], fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
    #             ColorJitter_func = lambda: x, self.ColorJitter(x, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
    #         static_rgb = np.reshape(np.array(rgb['rgb_static']), (-1, 3))
    #         gripper_rgb =  np.reshape(np.array(rgb['rgb_gripper']), (-1, 3))
    #         pcd_rgb = np.concatenate([static_rgb, gripper_rgb], axis=0)
    #         pcd_rgb = pcd_rgb/255
    #         pcd = self.vfe_generator.generate(cloud[:, :3], pcd_rgb)
    #         if 0:
    #             _pcd = o3d.geometry.PointCloud()
    #             point_tmp, rgb_tmp = self.vfe_generator.decode_occupied_grid_with_range(pcd)
    #             _pcd.points = o3d.utility.Vector3dVector(point_tmp)
    #             _pcd.colors = o3d.utility.Vector3dVector(rgb_tmp)
    #             o3d.io.write_point_cloud("tmp.pcd", _pcd)
    #             _pcd.points = o3d.utility.Vector3dVector(static_pcd)
    #             _pcd.colors = o3d.utility.Vector3dVector(static_rgb/255)
    #             o3d.io.write_point_cloud("tmp1.pcd", _pcd)
    #             _pcd.points = o3d.utility.Vector3dVector(gripper_pcd)
    #             _pcd.colors = o3d.utility.Vector3dVector(gripper_rgb/255)
    #             o3d.io.write_point_cloud("tmp2.pcd", _pcd)
    #         calib['static_extrinsic_matrix'] = static_extrinsic_matrix * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    #         calib['static_intrinsic_matrix'] = calib['rgb_static']['intrinsic_matrix']
    #         calib['static_distCoeffs_matrix'] = calib['rgb_static']['distCoeffs_matrix']
    #         calib['gripper_extrinsic_matrix'] = gripper_extrinsic_matrix * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    #         calib['gripper_intrinsic_matrix'] = calib['rgb_gripper']['intrinsic_matrix']
    #         calib['gripper_distCoeffs_matrix'] = calib['rgb_gripper']['distCoeffs_matrix']
    #         calib['state_matrix'] = ep['state_matrix'] * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    #         calib['static_fov'] = np.array([cam_config['static']['fov'], cam_config['static']['height'], cam_config['static']['width']])
    #         calib['gripper_fov'] = np.array([cam_config['gripper']['fov'], cam_config['gripper']['height'], cam_config['gripper']['width']])
    #         rgbs.append(rgb)
    #         pcds.append(pcd)
    #         calibs.append(calib)

    #     def _func(ep, ColorJitter_func=None):
    #         rgb = {}
    #         calib = ep['calib']
    #         cam_config = ep['cam_config']
    #         static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
    #         gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']
    #         static_cam = cam(static_extrinsic_matrix, cam_config['static']['height'], cam_config['static']['width'], cam_config['static']['fov'])
    #         gripper_cam = cam(gripper_extrinsic_matrix, cam_config['gripper']['height'], cam_config['gripper']['width'], cam_config['gripper']['fov'])
    #         static_pcd = deproject(static_cam, ep['depth_static'], homogeneous=False, sanity_check=False).transpose(1, 0)
    #         gripper_pcd = deproject(gripper_cam, ep['depth_gripper'], homogeneous=False, sanity_check=False).transpose(1, 0)
    #         cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)
    #         rgb['rgb_static'] = ep['rgb_static'] # Image.fromarray(ep['rgb_static'])
    #         rgb['rgb_gripper'] = ep['rgb_gripper'] # Image.fromarray(ep['rgb_gripper'])
    #         if ColorJitter_func is not None:
    #             rgb['rgb_static'], *_ = ColorJitter_func(rgb['rgb_static'])
    #             rgb['rgb_gripper'], *_ = ColorJitter_func(rgb['rgb_gripper'])
    #         static_rgb = np.reshape(np.array(rgb['rgb_static']), (-1, 3))
    #         gripper_rgb =  np.reshape(np.array(rgb['rgb_gripper']), (-1, 3))
    #         pcd_rgb = np.concatenate([static_rgb, gripper_rgb], axis=0)
    #         pcd_rgb = pcd_rgb/255
    #         pcd = self.vfe_generator.generate(cloud[:, :3], pcd_rgb)
    #         if 0:
    #             _pcd = o3d.geometry.PointCloud()
    #             point_tmp, rgb_tmp = self.vfe_generator.decode_occupied_grid_with_range(pcd)
    #             _pcd.points = o3d.utility.Vector3dVector(point_tmp)
    #             _pcd.colors = o3d.utility.Vector3dVector(rgb_tmp)
    #             o3d.io.write_point_cloud("tmp.pcd", _pcd)
    #             _pcd.points = o3d.utility.Vector3dVector(static_pcd)
    #             _pcd.colors = o3d.utility.Vector3dVector(static_rgb/255)
    #             o3d.io.write_point_cloud("tmp1.pcd", _pcd)
    #             _pcd.points = o3d.utility.Vector3dVector(gripper_pcd)
    #             _pcd.colors = o3d.utility.Vector3dVector(gripper_rgb/255)
    #             o3d.io.write_point_cloud("tmp2.pcd", _pcd)
    #         calib['static_extrinsic_matrix'] = static_extrinsic_matrix * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    #         calib['static_intrinsic_matrix'] = calib['rgb_static']['intrinsic_matrix']
    #         calib['static_distCoeffs_matrix'] = calib['rgb_static']['distCoeffs_matrix']
    #         calib['gripper_extrinsic_matrix'] = gripper_extrinsic_matrix * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    #         calib['gripper_intrinsic_matrix'] = calib['rgb_gripper']['intrinsic_matrix']
    #         calib['gripper_distCoeffs_matrix'] = calib['rgb_gripper']['distCoeffs_matrix']
    #         calib['state_matrix'] = ep['state_matrix'] * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    #         calib['static_fov'] = np.array([cam_config['static']['fov'], cam_config['static']['height'], cam_config['static']['width']])
    #         calib['gripper_fov'] = np.array([cam_config['gripper']['fov'], cam_config['gripper']['height'], cam_config['gripper']['width']])
            
    #         return rgb, pcd, calib

    #     partial_process_sample = functools.partial(_func, ColorJitter_func=ColorJitter_func)
    #     with ThreadPoolExecutor() as executor:
    #         results = list(executor.map(partial_process_sample, ['text', 'occ', 'static', 'gripper']))
    #     for rgb, pcd, calib in results:
    #         rgbs.append(rgb)
    #         pcds.append(pcd)
    #         calibs.append(calib)
    #     episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
    #     episode.update({key: np.stack([calib[key] for calib in calibs]) for key in keys_calib})
    #     episode.update({key: [rgb[key] for rgb in rgbs] for key in keys_rgb}) # episode.update({key: np.stack([rgb[key] for rgb in rgbs]) for key in keys_rgb})
    #     episode['pcd'] = np.stack(pcds)
        
    #     return episode

class DiskCalvinDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task,
                self.ep_start_end_ids
            ) = self._build_file_indices_lang(self.abs_datasets_dir) 
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)
        
        if self.env_resample and 'training' in str(self.abs_datasets_dir):
            self.scene_info = np.load(
                    f'{self.abs_datasets_dir}/scene_info.npy',
                    allow_pickle=True
                ).item()
            
        if 'task_D_D' in str(self.abs_datasets_dir):
            pass
        else:
            self.data_path_dic = {}
            for data_path in self.data_path_list:
                self.data_path_dic[data_path] = {}
                files = glob.glob(f"{data_path}/training_npz_pcd_new/*/episode_*")
                for file in files:
                    self.data_path_dic[data_path][file.split('/')[-1]] = file

    def _get_episode_name_split(self, file_idx: int, data_path) -> Path:

        if len(self.data_path_list)>0:
            #data_path = self.data_path_list[1]
            #data_path = random.choice(self.data_path_list)
            if self.validation:
                if 'task_D_D' in str(self.abs_datasets_dir):
                    return Path(f"{data_path}/validation/episode_{file_idx:0{7}d}.npz")
                else:
                    files = glob.glob(f"{data_path}/validation/*/episode_{file_idx:0{7}d}.npz")
                    file = files[0]
                    return Path(file)
            else:
                if 'task_D_D' in str(self.abs_datasets_dir):
                    return Path(f"{data_path}/training/episode_{file_idx:0{7}d}.npz")
                else:
                   
                    file = self.data_path_dic[data_path][f"episode_{file_idx:0{7}d}.npz"]
                    return Path(file)
                    
        if 'task_D_D' in str(self.abs_datasets_dir):
            return Path(
                f"{self.abs_datasets_dir}/episode_{file_idx:0{7}d}.npz"
            )
        else:
            files = glob.glob(f"{data_path}/training_npz_pcd_new/*/episode_{file_idx:0{7}d}.npz")
            file = files[0]
            return Path(file)

    def load_pcd(self,pcd_name):

        return o3d.io.read_point_cloud(str(pcd_name))

    def _get_episode(self, file_idx, data_path):

        episode = self.load_file(self._get_episode_name_split(file_idx, data_path))
        episode = {key: episode[key] for key in episode}
        state_matrix = self.load_file(self._get_episode_name_split(file_idx, self.state_matrixs_path))
        state_matrix = state_matrix['calib'].item()
        state_matrix = state_matrix['rgb_gripper']['extrinsic_matrix']

        episode['rgb_static'] = Image.fromarray(episode['rgb_static'])
        episode['rgb_gripper'] = Image.fromarray(episode['rgb_gripper'])
        episode['state_matrix'] = state_matrix
        episode['calib'] = episode['calib'].item()
        episode['cam_config'] = episode['cam_config'].item()
        return episode

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        
        # end_idx = self.episode_lookup[idx]+1 # 注意self.episode_lookup[idx]返回的是索引值，range()的话需要+1
        # start_idx_eposide, end_idx_eposide = self.ep_start_end_ids[self.lang_lookup[idx]]
        # start_idx = max(end_idx - window_size, start_idx_eposide)
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size

        if self.env_resample:
            if "task_ABC_D" in  str(self.abs_datasets_dir) :
                if ("calvin_scene_B" in self.scene_info and
                    start_idx <= self.scene_info["calvin_scene_B"][1]):
                    scene = "B"
                elif ("calvin_scene_C" in self.scene_info and
                    start_idx <= self.scene_info["calvin_scene_C"][1]):
                    scene = "C"
                elif ("calvin_scene_A" in self.scene_info and
                    start_idx <= self.scene_info["calvin_scene_A"][1]):
                    scene = "A"
                else:
                    scene = "D"
                    
                task = self.lang_task[self.lang_lookup[idx]]
                
                if ('slider' in task) or ('lightbulb' in task):
                    if scene == "A":
                        pass
                    else:
                        return 0
 
        # keys_calib = ['static_extrinsic_matrix','static_intrinsic_matrix','static_distCoeffs_matrix',
        #               'gripper_extrinsic_matrix','gripper_intrinsic_matrix','gripper_distCoeffs_matrix','state_matrix'] 
        keys_rgb = ['rgb_static','rgb_gripper'] 
        data_path = random.choice(self.data_path_list)
        if self.data_tasks_groups is not None:
            task = self.lang_task[self.lang_lookup[idx]]
            for i, data_tasks_group in enumerate(self.data_tasks_groups):
                if task in data_tasks_group:
                    data_path = self.data_path_list[i]
                    continue
        
        # episodes = [
        #     self.load_file(self._get_episode_name_split(file_idx, data_path))
        #     for file_idx in range(start_idx, end_idx)
        # ]
        # state_matrixs = [
        #     self.load_file(self._get_episode_name_split(file_idx, self.state_matrixs_path))
        #     for file_idx in range(start_idx, end_idx)
        # ]
        episodes = [self._get_episode(file_idx, data_path) for file_idx in range(start_idx, end_idx)]

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64, me
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"] 
        lang_lookup = []
        partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.only_single_task and "rotate_blue_block_right" not in lang_task[i]: continue # 仅测试使用
            if self.partial_data:
                if (start_idx, end_idx) not in partial_st_ed_list:
                    continue
            if self.pretrain: 
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
            # for idx in range(start_idx+1, end_idx): # YF：end_idx表示长度，而不是下标索引,注意此处start_idx+1是为了与window_size+1对应，生成下一张图片
                if cnt % self.skip_frames == 0: 
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task, ep_start_end_ids

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()
        assert False
        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)


class DiskMetaWorldDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        
        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, zip_file, frames):
        pos, grasp = np.split(frames[file_idx]['action'], [3]) # YF：pos乘了40变为了[-1,1]
        action = np.concatenate([pos, np.zeros_like(pos), [-1 if grasp>0 else 1]]) # grasp:[-0.5,0.5],0.5是关，期望-1是关

        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #     rgb = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "corner2", f'image_save_{file_idx}.jpg')))
        #     rgb2 = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "behindGripper", f'image_save_{file_idx}.jpg')))
        #     rgb_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "corner2", f'depth_save_{file_idx}.npy')))
        #     rgb2_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "behindGripper", f'depth_save_{file_idx}.npy')))
        #     calib = np.load(load_from_zip(zip_ref, os.path.join('rgb', "corner2", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        #     calib2 = np.load(load_from_zip(zip_ref, os.path.join('rgb', "behindGripper", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        rgb = Image.open(os.path.join(zip_file, os.path.join('rgb', "corner2", f'image_save_{file_idx}.jpg')))
        rgb2 = Image.open(os.path.join(zip_file, os.path.join('rgb', "behindGripper", f'image_save_{file_idx}.jpg')))
        rgb_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "corner2", f'depth_save_{file_idx}.npy')))
        rgb2_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "behindGripper", f'depth_save_{file_idx}.npy')))
        calib = np.load(os.path.join(zip_file, os.path.join('rgb', "corner2", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        calib2 = np.load(os.path.join(zip_file, os.path.join('rgb', "behindGripper", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM) # rgb = np.flip(rgb, axis=0) # 注意此处需要添加flip
        rgb2 = rgb2.transpose(Image.FLIP_TOP_BOTTOM) # rgb2 = np.flip(rgb2, axis=0)
        def depthimg2Meters(depth, cam):
            extent =cam['cam_config']['extent']
            near = cam['cam_config']['nearval'] * extent
            far = cam['cam_config']['farval'] * extent
            image = near / (1 - depth * (1 - near / far))
            return image
        rgb_depth = np.flip(rgb_depth, axis=0)
        rgb_depth = depthimg2Meters(rgb_depth, calib)
        rgb2_depth = np.flip(rgb2_depth, axis=0)
        rgb2_depth = depthimg2Meters(rgb2_depth, calib2)
        calib['extrinsic_matrix'] = np.linalg.inv(calib['extrinsic_matrix']) * np.array([[1,1,1,1], [-1,-1,-1,-1], [-1,-1,-1,-1], [1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
        calib2['extrinsic_matrix'] = np.linalg.inv(calib2['extrinsic_matrix']) * np.array([[1,1,1,1], [-1,-1,-1,-1], [-1,-1,-1,-1], [1,1,1,1]])
        T_translate = np.array([[1, 0, 0, 0],   # [-0.5038039  -0.10849035  0.12061001  1.        ] [0.48056049 0.41590198 0.60125949 1.        ]
                                [0, 1, 0, 0.25],
                                [0, 0, 1, -0.25],
                                [0, 0, 0, 1]])
        calib['extrinsic_matrix'] = np.dot(calib['extrinsic_matrix'], T_translate)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], T_translate) # 为了与calvin点云坐标对齐

        return {
            "rgb_static": rgb, # np.array(rgb),
            "rgb_gripper": rgb2, # np.array(rgb2),
            "rel_actions": action,  # action_scale: float = 1.0 / 100
            "robot_obs": frames[file_idx]['obs'],
            "scene_obs": frames[file_idx]['obs'],
            "depth_static": rgb_depth,
            "depth_gripper": rgb2_depth,
            "calib": {  "rgb_static": calib,
                        "rgb_gripper": calib2},
            "cam_config": { 'static': calib['cam_config'],
                            'gripper': calib2['cam_config']},
            "state_matrix": calib2['extrinsic_matrix'] # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # zip_file, end_idx, video_len, n_task = self.episode_lookup[idx]
        # end_idx += 1 # 注意返回的是索引值，range()的话需要+1
        zip_file, start_idx, video_len, n_task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        zip_file = os.path.join(self.abs_datasets_dir, zip_file)
        # start_idx = max(end_idx - window_size, 0)

        zip_file = os.path.splitext(zip_file)[0]
        with open(os.path.join(zip_file, 'param/param.pickle'), 'rb') as file:
            frames = pickle.load(file)
            episodes = [self._get_episode(file_idx, zip_file, frames) for file_idx in range(start_idx, end_idx)]
        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #     frames = pickle.load(load_from_zip(zip_ref, 'param/param.pickle'))
        #     episodes = [self._get_episode(file_idx, zip_file, frames) for file_idx in range(start_idx, end_idx)]
        
        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[n_task]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        json_path = os.path.join(abs_datasets_dir, "train_metaworld.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        for data_slice in data:
            zip_file = data_slice[0]
            video_len = data_slice[1]
            if self.only_single_task and "basketball-v2-goal-observable" not in zip_file: continue
            assert video_len <= 500
            lang_ann[data_slice[2]] = data_slice[3]
            for start_frame_index in range(0, video_len+1-self.min_window_size):
            # for start_frame_index in range(0+1, video_len): # YF: 看calvin
                episode_lookup.append(
                    (zip_file, start_frame_index, video_len, data_slice[2]))
        return episode_lookup, lang_ann
        

class DiskLiberoDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, demo_path):
        
        def rotMatList2NPRotMat(rot_mat_arr):
            np_rot_arr = np.array(rot_mat_arr)
            np_rot_mat = np_rot_arr.reshape((3, 3))
            return np_rot_mat
        def posRotMat2Mat(pos, rot_mat):
            t_mat = np.eye(4)
            t_mat[:3, :3] = rot_mat
            t_mat[:3, 3] = np.array(pos)
            return t_mat

        file_path = demo_path / f"{file_idx}.npz"
        episode = np.load(file_path, allow_pickle=True)
        action=np.concatenate([episode['action'][:6], [-1 if episode['action'][-1]>0 else 1]]) # YF: libero episode['action'][-1]为-1/1，其中1表示关; 
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        agentview = episode['agentview'].item()
        robot0_eye_in_hand = episode['robot0_eye_in_hand'].item()
        agentview['calib']['extrinsic_matrix'] = np.linalg.inv(agentview['calib']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])  # 主要原因是deproject，对YZ加了一个符号
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.linalg.inv(robot0_eye_in_hand['calib']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        base_pose = posRotMat2Mat(episode['base_p'], rotMatList2NPRotMat(episode['base_r']))
        agentview['calib']['extrinsic_matrix'] = np.dot(agentview['calib']['extrinsic_matrix'], base_pose)
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.dot(robot0_eye_in_hand['calib']['extrinsic_matrix'], base_pose)
        # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.3],    # 
            [0, 1, 0, 0.0],     # 
            [0, 0, 1, -0.1],    # [-0.24398557 -0.43166057  0.01677953  1.        ] [0.86626239 0.57459427 0.90260449 1.        ]  
            [0, 0, 0, 1] 
        ])
        agentview['calib']['extrinsic_matrix'] = np.dot(agentview['calib']['extrinsic_matrix'], T_translate)
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.dot(robot0_eye_in_hand['calib']['extrinsic_matrix'], T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [-1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        agentview['calib']['extrinsic_matrix']=np.dot(agentview['calib']['extrinsic_matrix'], R)
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.dot(robot0_eye_in_hand['calib']['extrinsic_matrix'], R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {
            "rgb_static": Image.fromarray(agentview['image']),
            "rgb_gripper": Image.fromarray(robot0_eye_in_hand['image']),
            "rel_actions": action,
            "robot_obs": np.zeros(20), # 由于没有保存，且没有使用，直接复制为0了。
            "scene_obs": np.zeros(20),
            "depth_static": agentview['depth'],
            "depth_gripper": robot0_eye_in_hand['depth'],
            "calib": {  'rgb_static': agentview['calib'],
                        'rgb_gripper': robot0_eye_in_hand['calib']},
            "cam_config": { "static": {"height": 256, "width": 256, "fov": 45},
                            "gripper": {"height": 256, "width": 256, "fov": 75}},
            "state_matrix": robot0_eye_in_hand['calib']['extrinsic_matrix']  # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # task, demo_id, end_idx = self.episode_lookup[idx]
        # end_idx+=1
        # demo_path = self.abs_datasets_dir / task / f"demo_{demo_id}"
        # start_idx = max(end_idx - window_size, 0)
        task, demo_id, start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        demo_path = self.abs_datasets_dir / task / f"demo_{demo_id}"
        episodes = [self._get_episode(file_idx, demo_path) for file_idx in range(start_idx, end_idx)]

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[task]
            # if self.text_aug:
            #     task = self.lang_task[self.lang_lookup[idx]]
            #     enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
            #     episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup, lang_ann = [], {}
        task_emb_list = np.load(abs_datasets_dir/'emb.npz', allow_pickle=True)
        
        _task_list = os.listdir(abs_datasets_dir)
        for task in _task_list:
            data_dir = abs_datasets_dir / task
            if not data_dir.is_dir(): continue # 样本均在文件夹内，假如非文件夹则无效
            if self.only_single_task and task not in ['task_86', 'task_87', 'task_88', 'task_89'] : continue # 调试使用
            lang_ann[task] = task_emb_list['arr_0'][int(task.split('_')[-1])]
            demo_num = len(os.listdir(data_dir))
            for i in range(demo_num):
                step_num = len(os.listdir(data_dir / f"demo_{i}"))
                for j in range(step_num+1 - self.min_window_size):
                # for j in range(0+1, step_num):
                    episode_lookup.append((task, i, j))

        return episode_lookup, lang_ann
        

class DiskRobocasaDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann,
                self.base_extrinsics
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, zip_file, frames, base_extrinsic):

        action = frames[file_idx]['action'][:7] # 注意这里移动12个值，前6个是爪子位姿，第7个是爪子张合1表示关，第8-10是移动，11是躯干，12是底盘移动还是夹爪
        action[-1] = -1 if action[-1] > 0 else 1  # 1表示关
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        # ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand", "robot0_agentview_center", "robot0_frontview"],
        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #     rgb = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'image_save_' + str(file_idx) + ".jpg")))
        #     rgb2 = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg")))
        #     rgb_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'depth_save_' + str(file_idx) + ".npy")))
        #     rgb2_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy")))
        #     calib = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        #     calib2 = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        rgb = Image.open(os.path.join(zip_file, os.path.join('rgb', "robot0_agentview_left", 'image_save_' + str(file_idx) + ".jpg")))
        rgb2 = Image.open(os.path.join(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg")))
        rgb_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_agentview_left", 'depth_save_' + str(file_idx) + ".npy")))
        rgb2_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy")))
        calib = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_agentview_left", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        calib2 = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM) # rgb = np.flip(np.array(rgb), axis=0 # 注意此处需要添加flip
        rgb2 = rgb2.transpose(Image.FLIP_TOP_BOTTOM) # rgb2 = np.flip(np.array(rgb2), axis=0)
        def depthimg2Meters(depth, cam):
            extent =cam['cam_config']['extent']
            near = cam['cam_config']['nearval'] * extent
            far = cam['cam_config']['farval'] * extent
            image = near / (1 - depth * (1 - near / far))
            return image
        rgb_depth = np.flip(rgb_depth, axis=0)
        rgb_depth = depthimg2Meters(rgb_depth, calib)
        rgb2_depth = np.flip(rgb2_depth, axis=0)
        rgb2_depth = depthimg2Meters(rgb2_depth, calib2)
        calib['extrinsic_matrix'] = np.linalg.inv(calib['extrinsic_matrix']) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        calib2['extrinsic_matrix'] = np.linalg.inv(calib2['extrinsic_matrix']) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        calib['extrinsic_matrix'] = np.dot(calib['extrinsic_matrix'], base_extrinsic)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], base_extrinsic)
        # 平移矩阵 T_translate single_stage_kitchen_pnp_PnPMicrowaveToCounter 0.6-1.0
        T_translate = np.array([
            [1, 0, 0, 0.3], #  [-0.81904223 -1.35147707  0.70431699  1.        ] [0.8594698  0.75282846 1.83341838 1.        ]
            [0, 1, 0, 0.0], # 
            [0, 0, 1, 0.7], # 
            [0, 0, 0, 1]
        ])
        calib['extrinsic_matrix'] = np.dot(calib['extrinsic_matrix'], T_translate)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [-1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        calib['extrinsic_matrix']=np.dot(calib['extrinsic_matrix'], R)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {
            "rgb_static": rgb,
            "rgb_gripper": rgb2,
            "rel_actions": action,
            "robot_obs": frames[file_idx]['states'],
            "scene_obs": frames[file_idx]['states'],
            "depth_static": rgb_depth,
            "depth_gripper": rgb2_depth,
            "calib": {  "rgb_static": calib,
                        "rgb_gripper": calib2},
            "cam_config":{  "static": calib['cam_config'],
                            "gripper": calib2['cam_config']},
            "state_matrix": calib2['extrinsic_matrix'] # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # zip_file, end_idx, video_len, n_task = self.episode_lookup[idx]
        # end_idx += 1 # YF: 看calvin
        # start_idx = max(end_idx - window_size, 0)
        zip_file, start_idx, video_len, n_task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        base_extrinsic = self.base_extrinsics[zip_file]

        with open(os.path.join(os.path.splitext(os.path.join(self.abs_datasets_dir, zip_file))[0], 'param/param.pickle'), 'rb') as file:
            frames = pickle.load(file)
            episodes = [self._get_episode(file_idx, os.path.splitext(os.path.join(self.abs_datasets_dir, zip_file))[0], frames, base_extrinsic) for file_idx in range(start_idx, end_idx)]
        # with zipfile.ZipFile(os.path.join(self.abs_datasets_dir, zip_file), 'r') as zip_ref:
        #     frames = pickle.load(load_from_zip(zip_ref, 'param/param.pickle'))
        #     episodes = [self._get_episode(file_idx, os.path.join(self.abs_datasets_dir, zip_file), frames, base_extrinsic) for file_idx in range(start_idx, end_idx)]
        
        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[zip_file]
            # if self.text_aug:
            #     task = self.lang_task[self.lang_lookup[idx]]
            #     enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
            #     episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, base_extrinsics, episode_lookup = {}, {}, []
        json_path = os.path.join(abs_datasets_dir, "train_metaworld.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        for data_slice in data:
            if 'navigat' in data_slice[2]: continue
            if self.only_single_task and 'CloseDrawer' not in data_slice[2]: continue # 仅调试使用
            zip_file = data_slice[0]
            video_len = data_slice[1]
            # if "basketball-v2-goal-observable" not in zip_file: continue
            # assert video_len <= 500
            lang_ann[zip_file] = data_slice[3]
            base_extrinsics[zip_file] = np.array(data_slice[4])
            for start_frame_index in range(0, video_len+1-self.min_window_size):
            # for start_frame_index in range(0+1, video_len): # YF: 看calvin
                episode_lookup.append(
                    (zip_file, start_frame_index, video_len, data_slice[2]))
        return episode_lookup, lang_ann, base_extrinsics


class DiskRoboMimicDataset(BaseMultiDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.tasks = ["lift", "can", "square"] # transport为双臂，["lift", "can", "square", "transport", "tool_hang"]
        self.lang_ann = {
            "lift" : "The robot arm must lift a small cube.",
            "can" : "The robot must place a coke can from a large bin into a smaller target bin.",
            "square" : "The robot must pick a square nut and place it on a rod.",
            "transport" : "Two robot arms must transfer a hammer from a closed container on a shelf to a target bin on another shelf.",
            "tool_hang" : "Insert the hook into the base, assemble the frame, and hang the wrench on the hook.",
        }
        if self.with_lang:
            (
                self.episode_lookup
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx, frames, cam_infos): 
        # action: [-1.         -1.         -1.         -0.55634028 -1.         -1. -1.        ] [1.         1.         1.         0.72973686 0.45003703 1. 1.        ]
        action = frames["actions"][file_idx][()]
        action[-1] = -1 if action[-1] > 0 else 1 # robomimic中1表示关
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        rgb_static = frames["obs"]['agentview_image'][file_idx][()] # 256 256 3
        rgb_gripper = frames["obs"]['robot0_eye_in_hand_image'][file_idx][()] # 256 256 3
        depth_static = np.squeeze(frames["obs"]['agentview_depth'][file_idx][()])
        depth_gripper = np.squeeze(frames["obs"]['robot0_eye_in_hand_depth'][file_idx][()])
        camera_info = cam_infos[f"{file_idx}"]

        static_extrinsic_matrix = np.linalg.inv(np.array(camera_info["agentview"]['extrinsics'])) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])   #注意这里没有求逆
        gripper_extrinsic_matrix = np.linalg.inv(np.array(camera_info['robot0_eye_in_hand']['fix_extrinsics'])) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
        T_translate = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.4], # [-0.17879518 -0.40093808  0.90953124  1.        ] [0.33830822 0.33967275 1.29464515 1.        ]
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [-1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        calib = {}
        calib['rgb_static'] = {} 
        calib['rgb_static']['extrinsic_matrix'] = static_extrinsic_matrix # 4 4 
        calib['rgb_static']['intrinsic_matrix'] = np.array(camera_info["agentview"]["intrinsics"]) # 3 3
        calib['rgb_static']['distCoeffs_matrix'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]) # 8

        calib['rgb_gripper'] = {}
        calib['rgb_gripper']['extrinsic_matrix'] = gripper_extrinsic_matrix # 4 4 
        calib['rgb_gripper']['intrinsic_matrix'] = np.array(camera_info['robot0_eye_in_hand']["intrinsics"]) # 3 3
        calib['rgb_gripper']['distCoeffs_matrix'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]) # 8

        def calculate_fov(intrinsic_matrix, image_width):
            fx = intrinsic_matrix[0, 0]
            fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
            return fov
        cam_config = {}
        cam_config['static'] = {}
        cam_config['static']['height'] = rgb_static.shape[0]
        cam_config['static']['width'] = rgb_static.shape[1]
        cam_config['static']['fov'] = calculate_fov(calib['rgb_static']['intrinsic_matrix'], rgb_static.shape[0])
        cam_config['gripper'] = {}
        cam_config['gripper']['height'] = rgb_gripper.shape[0]
        cam_config['gripper']['width'] = rgb_gripper.shape[1]
        cam_config['gripper']['fov'] = calculate_fov(calib['rgb_gripper']['intrinsic_matrix'], rgb_gripper.shape[0])
        # joints = frames["obs"]["robot0_joint_pos"][file_idx]
        return {
            "rgb_static": Image.fromarray(rgb_static),
            "rgb_gripper": Image.fromarray(rgb_gripper),
            "depth_static": depth_static,
            "depth_gripper": depth_gripper, # 单位m
            "rel_actions": action,
            "robot_obs": np.zeros(20),
            "scene_obs": np.zeros(20),
            "calib": calib,
            "cam_config": cam_config,
            "state_matrix": gripper_extrinsic_matrix  # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        dataset_file, episode_id, start_idx, episode_length, task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        
        with h5py.File(dataset_file, "r") as data:       
            def load_h5_data(data):
                out = dict()
                for k in data.keys():
                    if isinstance(data[k], h5py.Dataset): out[k] = data[k][:]
                    else: out[k] = load_h5_data(data[k])
                return out
            frames = data["data"][episode_id]
            cam_infos = json.loads(frames.attrs['camera_info_realtime'])
            # frames = load_h5_data(frames)  # 暂时不使用
            episodes = [self._get_episode(file_idx, frames, cam_infos) for file_idx in range(start_idx, end_idx)]

            episode = self._prepare_eposides(episodes)

            if self.with_lang:
                episode["language"] = self.lang_ann[task]
                if self.text_aug:
                    task = self.lang_task[self.lang_lookup[idx]]
                    enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                    episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
        """
        episode_lookup = []
        # 加载不同任务
        for task in self.tasks:
            if self.only_single_task and "lift" not in task: continue
            dataset_file = os.path.join(abs_datasets_dir, task, "ph", "depth.hdf5")
            data = h5py.File(dataset_file, "r")
            episodes = data["data"]             # 记录每一条轨迹的信息
            for i in range(len(episodes)):             # 构建episode_lookup
                episode_length = len(episodes["demo_{0}".format(i)]["actions"])
                for start_frame_index in range(0, episode_length + 1 - self.min_window_size):
                    episode_lookup.append((dataset_file, "demo_{0}".format(i), start_frame_index, episode_length, task))
            data.close()
        return episode_lookup # zyf
        

class DiskManiSkill2Dataset(BaseMultiDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    # def _get_episode(self, file_idx, frames): 
    #     from scipy.spatial.transform import Rotation as R
    #     def base_pose_to_matrix(base_pose):
    #         position = base_pose[:3] # 提取位置和四元数
    #         orientation = base_pose[3:]
    #         rotation = R.from_quat(orientation).as_matrix() # 将四元数转换为旋转矩阵
    #         transform_matrix = np.eye(4) # 创建4x4的同质变换矩阵
    #         transform_matrix[:3, :3] = rotation
    #         transform_matrix[:3, 3] = position
    #         return transform_matrix

    #     action = frames["actions"][file_idx][()]
    #     action = np.array([action[1], action[0], -action[2], action[4], action[3], -action[5], action[6]])  # 对齐坐标系[1, 0,-2,4,3,-5,6]

    #     rgb_static = frames["obs"]["image"]["base_camera"]["rgb"][file_idx][()]
    #     rgb_gripper = frames["obs"]["image"]["hand_camera"]["rgb"][file_idx][()]
    #     depth_static = np.squeeze(frames["obs"]["image"]["base_camera"]["depth"][file_idx][()])
    #     depth_gripper = np.squeeze(frames["obs"]["image"]["hand_camera"]["depth"][file_idx][()])
    #     static_extrinsic_matrix = frames["obs"]["camera_param"]["base_camera"]['extrinsic_cv'][file_idx][()] * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])   # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
    #     gripper_extrinsic_matrix = frames["obs"]["camera_param"]["hand_camera"]['extrinsic_cv'][file_idx][()] * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    #     # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
    #     base_pose = base_pose_to_matrix(frames['obs']['agent']['base_pose'][file_idx][()])
    #     static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_pose)
    #     gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_pose)
    #     # 平移矩阵 T_translate 待修改
    #     T_translate = np.array([
    #         [1, 0, 0, 0.3],
    #         [0, 1, 0, 0.0],
    #         [0, 0, 1, 0.0], # [-0.2615632  -0.79399014 -1.17142143  1. ] [8.50e-01 7.642e-01 8.9469e-04 1.00e+00]
    #         [0, 0, 0, 1]
    #     ])
    #     static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
    #     gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
    #     R = np.array([
    #         [0, 1, 0, 0],
    #         [1,  0, 0, 0],
    #         [0,  0, -1, 0],
    #         [0,  0, 0, 1],
    #     ])
    #     static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
    #     gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往右为Y轴；往下为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
    #     calib = {'rgb_static': {'extrinsic_matrix': static_extrinsic_matrix,         
    #                             'intrinsic_matrix': frames["obs"]["camera_param"]["base_camera"]["intrinsic_cv"][file_idx][()],
    #                             'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
    #             'rgb_gripper': {'extrinsic_matrix': gripper_extrinsic_matrix,
    #                             'intrinsic_matrix': frames["obs"]["camera_param"]["hand_camera"]["intrinsic_cv"][file_idx][()],
    #                             'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}}
    #     def calculate_fov(intrinsic_matrix, image_width):
    #         fx = intrinsic_matrix[0, 0]
    #         fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
    #         return fov
    #     cam_config = {'static':{'height': rgb_static.shape[0],
    #                             'width': rgb_static.shape[1],
    #                             'fov': calculate_fov(frames["obs"]["camera_param"]["base_camera"]["intrinsic_cv"][file_idx][()], rgb_static.shape[0])}, #90
    #                  'gripper':{'height': rgb_gripper.shape[0],
    #                             'width': rgb_gripper.shape[1],
    #                             'fov': calculate_fov(frames["obs"]["camera_param"]["hand_camera"]["intrinsic_cv"][file_idx][()], rgb_gripper.shape[0])}}
    #     # joints = frames["obs"]["agent"]["base_pose"][file_idx]
        
    #     return {
    #         "rgb_static": Image.fromarray(rgb_static),
    #         "rgb_gripper": Image.fromarray(rgb_gripper),
    #         "depth_static": depth_static / 1000,
    #         "depth_gripper": depth_gripper/ 1000, # 单位mm
    #         "rel_actions": action,
    #         "robot_obs": np.zeros(20),
    #         "scene_obs": np.zeros(20),
    #         "calib": calib,
    #         "cam_config": cam_config,
    #         "state_matrix": gripper_extrinsic_matrix,  # 注意此处是爪子矩阵
    #     }

    def _get_episode(self, start_idx, end_idx, frames): 
        from scipy.spatial.transform import Rotation as R
        def base_pose_to_matrix(base_pose):
            position = base_pose[:, :3] # 提取位置和四元数
            orientation = base_pose[:, 3:]
            rotation = R.from_quat(orientation).as_matrix() # 将四元数转换为旋转矩阵
            transform_matrix = np.eye(4) # 创建4x4的同质变换矩阵
            transform_matrix = np.tile(transform_matrix, (13, 1, 1))
            transform_matrix[:, :3, :3] = rotation
            transform_matrix[:, :3, 3] = position
            return transform_matrix

        action = frames["actions"][start_idx:end_idx][()]
        action = np.hstack([action[:, 1:2], action[:, 0:1], -action[:, 2:3], action[:, 4:5], action[:, 3:4], -action[:, 5:6], action[:, 6:7]])  # 对齐坐标系[1, 0,-2,4,3,-5,6]

        rgb_static = frames["obs"]["image"]["base_camera"]["rgb"][start_idx:end_idx][()]
        rgb_gripper = frames["obs"]["image"]["hand_camera"]["rgb"][start_idx:end_idx][()]
        depth_static = np.squeeze(frames["obs"]["image"]["base_camera"]["depth"][start_idx:end_idx][()])
        depth_gripper = np.squeeze(frames["obs"]["image"]["hand_camera"]["depth"][start_idx:end_idx][()])
        if len(depth_static.shape) == 2:
            depth_static = depth_static[None]
            depth_gripper = depth_gripper[None]
        static_extrinsic_matrix = frames["obs"]["camera_param"]["base_camera"]['extrinsic_cv'][start_idx:end_idx][()] * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])   # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
        gripper_extrinsic_matrix = frames["obs"]["camera_param"]["hand_camera"]['extrinsic_cv'][start_idx:end_idx][()] * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        base_pose = base_pose_to_matrix(frames['obs']['agent']['base_pose'][start_idx:end_idx][()])
        static_extrinsic_matrix = np.einsum('mij,mjk->mik', static_extrinsic_matrix, base_pose)
        gripper_extrinsic_matrix = np.einsum('mij,mjk->mik', gripper_extrinsic_matrix, base_pose)
        # 平移矩阵 T_translate 待修改
        T_translate = np.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.0], # [-0.2615632  -0.79399014 -1.17142143  1. ] [8.50e-01 7.642e-01 8.9469e-04 1.00e+00]
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, -1, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, R) # 往前为X轴；往右为Y轴；往下为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        static_intrinsic_matrix = frames["obs"]["camera_param"]["base_camera"]["intrinsic_cv"][start_idx:end_idx][()]
        gripper_intrinsic_matrix = frames["obs"]["camera_param"]["hand_camera"]["intrinsic_cv"][start_idx:end_idx][()]
        def calculate_fov(intrinsic_matrix, image_width):
            fx = intrinsic_matrix[0, 0]
            fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
            return fov
        cam_config = {'static':{'height': rgb_static.shape[1],
                                'width': rgb_static.shape[2],
                                'fov': calculate_fov(static_intrinsic_matrix[0], rgb_static.shape[1])}, #90
                     'gripper':{'height': rgb_gripper.shape[1],
                                'width': rgb_gripper.shape[2],
                                'fov': calculate_fov(gripper_intrinsic_matrix[0], rgb_gripper.shape[1])}}
        return [{   "rgb_static": Image.fromarray(rgb_static_),
                    "rgb_gripper": Image.fromarray(rgb_gripper_),
                    "depth_static": depth_static_ / 1000,
                    "depth_gripper": depth_gripper_ / 1000, # 单位mm
                    "rel_actions": action_,
                    "robot_obs": np.zeros(20),
                    "scene_obs": np.zeros(20),
                    "calib": {  'rgb_static': { 'extrinsic_matrix': static_extrinsic_matrix_,         
                                                'intrinsic_matrix': static_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                                'rgb_gripper': {'extrinsic_matrix': gripper_extrinsic_matrix_,
                                                'intrinsic_matrix': gripper_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}},
                    "cam_config": cam_config,
                    "state_matrix": gripper_extrinsic_matrix_,  # 注意此处是爪子矩阵
        }for (rgb_static_, rgb_gripper_, depth_static_, depth_gripper_, action_, static_extrinsic_matrix_, static_intrinsic_matrix_, gripper_extrinsic_matrix_, gripper_intrinsic_matrix_) in zip(rgb_static, rgb_gripper, depth_static, depth_gripper, action, static_extrinsic_matrix, static_intrinsic_matrix, gripper_extrinsic_matrix, gripper_intrinsic_matrix)]

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # dataset_file, episode_id, end_idx, episode_length, task = self.episode_lookup[idx]
        # end_idx +=1
        # data = h5py.File(dataset_file, "r")
        # frames = data[f"traj_{episode_id}"]
        # start_idx = max(end_idx - window_size, 0)
        dataset_file, episode_id, start_idx, episode_length, task = self.episode_lookup[idx]
        end_idx = start_idx + window_size

        with h5py.File(dataset_file, "r") as data:
            frames = data[f"traj_{episode_id}"]
            def load_h5_data(data):
                out = dict()
                for k in data.keys():
                    if isinstance(data[k], h5py.Dataset): out[k] = data[k][:]
                    else: out[k] = load_h5_data(data[k])
                return out
            # frames = load_h5_data(frames) # 暂时不使用
            # episodes = [self._get_episode(file_idx, frames)for file_idx in range(start_idx, end_idx)]
            episodes = self._get_episode(start_idx, end_idx, frames)

            episode = self._prepare_eposides(episodes)

            if self.with_lang:
                episode["language"] = self.lang_ann[task]
                if self.text_aug:
                    task = self.lang_task[self.lang_lookup[idx]]
                    enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                    episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        task_emb_list = np.load(os.path.join(abs_datasets_dir, 'maniskill2_lang.py.npy'), allow_pickle=True).item()
        abs_datasets_dir = os.path.join(abs_datasets_dir, "v0")

        def find_files_with_suffix(root_path, suffix):
            import glob
            search_pattern = os.path.join(root_path, '**', f'*{suffix}') # 构建搜索模式
            matching_files = glob.glob(search_pattern, recursive=True) # 在指定路径下递归搜索所有符合后缀的文件
            absolute_paths = [os.path.abspath(file) for file in matching_files] # 获取并返回这些文件的绝对路径
            return absolute_paths
        lang_ann, episode_lookup = {}, []
        h5_files = find_files_with_suffix(abs_datasets_dir, 'rgbd.pd_ee_delta_pose.h5')
        for h5_f in h5_files:
            task = [elem for elem in list(task_emb_list.keys()) if elem in h5_f]
            assert len(task)==1
            if "soft_body" in h5_f: continue # 暂时不使用soft_body，task是Excavate-v0/Fill-v0/Hang-v0
            if self.only_single_task and 'LiftCube' not in task[0]: continue # 仅调试使用
            lang_ann[task[0]] = task_emb_list[task[0]]

            json_path = h5_f.replace(".h5", ".json")
            json_data = json.load(open(json_path, 'rb'))
            episodes = json_data["episodes"]
            for i in range(len(episodes)):
                assert i == episodes[i]["episode_id"]
                episode_length = episodes[i]["elapsed_steps"]
                for start_frame_index in range(0, episode_length + 1 - self.min_window_size):
                # for start_frame_index in range(0+1, episode_length): # YF:看calvin
                    episode_lookup.append((h5_f, episodes[i]["episode_id"], start_frame_index, episode_length, task[0]))
                
        return episode_lookup, lang_ann


class DiskRlbenchDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, episodes_dir, frames):
        
        def angle_between_angles(a, b):
            diff = b - a
            return (diff + np.pi) % (2 * np.pi) - np.pi
        def to_relative_action(actions, robot_obs, max_pos=0.03, max_orn=0.2):
            rel_pos = actions[:3] - robot_obs[:3]
            rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
            rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
            rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
            gripper = actions[-1:]
            return np.concatenate([rel_pos, rel_orn, gripper])
        r_euler = np.array(pybullet.getEulerFromQuaternion(frames[file_idx].gripper_pose[3:7]))
        a_euler = np.array(pybullet.getEulerFromQuaternion(frames[file_idx+1].gripper_pose[3:7]))
        if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8 and (r_euler[0]==0 or a_euler[0]==0 or r_euler[2]==0 or a_euler[2]==0): # 发生了万向节死锁
            assert False, 'Maybe gimbal lock has occurred.'
        # if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8: # 大于0.8就发生了万向节死锁
        #     sr_euler = np.array(scipyR.from_quat(frames[file_idx].gripper_pose[3:7]).as_euler('xyz', degrees=False))
        #     sa_euler = np.array(scipyR.from_quat(frames[file_idx+1].gripper_pose[3:7]).as_euler('xyz', degrees=False))
        #     if np.abs(angle_between_angles(a_euler, r_euler) - angle_between_angles(sa_euler, sr_euler)).max()<0.01: # 并没有发生万向节死锁
        #         pass
        #     elif np.abs(angle_between_angles(sa_euler, sr_euler)).max() < 0.8: # 大于0.8就发生了万向节死锁
        #         r_euler = sr_euler
        #         a_euler = sa_euler
        #     else:
        #         if np.abs(angle_between_angles(sa_euler, r_euler)).max() < 0.8:
        #             a_euler = sa_euler
        #         elif np.abs(angle_between_angles(a_euler, sr_euler)).max() < 0.8:
        #             r_euler = sr_euler
        #         else:
        #             assert False, 'Maybe gimbal lock has occurred.'
        robot_obs_euler = np.concatenate([frames[file_idx].gripper_pose[:3], r_euler, [1 if frames[file_idx].gripper_open>0.5 else -1]]) # YF: gripper_open=0/1, 注意要求的是-1和1
        actions_euler = np.concatenate([frames[file_idx+1].gripper_pose[:3], a_euler, [1 if frames[file_idx+1].gripper_open>0.5 else -1]])
        action = to_relative_action(actions_euler, robot_obs_euler[:6], max_pos=0.03, max_orn=0.2)
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        rgb = Image.open(episodes_dir/f"front_rgb/{file_idx}.png")
        rgb2 = Image.open(episodes_dir/f"wrist_rgb/{file_idx}.png")
        def calculate_fov(intrinsic_matrix, image_width):
            fx = intrinsic_matrix[0, 0]
            fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
            return fov
        def image_to_float_array(image, scale_factor): 
            image_array = np.array(image)
            assert 2 <= image_array.ndim <= 3, "Image must be either 2D or 3D array."
            
            if image_array.ndim == 3 and image_array.shape[2] == 3:
                # RGB image: Convert to 24-bit integer using bitwise operations for better performance
                float_array = (image_array[:, :, 0].astype(np.uint32) << 16) + \
                            (image_array[:, :, 1].astype(np.uint32) << 8) + \
                            image_array[:, :, 2].astype(np.uint32)
            else:
                # Grayscale image or single channel image
                float_array = image_array.astype(np.float32)
            
            scaled_array = float_array / scale_factor
            return scaled_array
        static_cam_config = {"width": rgb.width, "height": rgb.height, "fov": calculate_fov(frames[file_idx].misc['front_camera_intrinsics'], rgb.height), "nearval": frames[file_idx].misc['front_camera_near'], "farval":  frames[file_idx].misc['front_camera_far']}
        gripper_cam_config = {"width": rgb2.width, "height": rgb2.height, "fov": calculate_fov(frames[file_idx].misc['wrist_camera_intrinsics'], rgb2.height), "nearval": frames[file_idx].misc['wrist_camera_near'], "farval":  frames[file_idx].misc['wrist_camera_far']}
        depth_static = image_to_float_array(Image.open(episodes_dir/f"front_depth/{file_idx}.png"), 16777215)
        depth_gripper = image_to_float_array(Image.open(episodes_dir/f"wrist_depth/{file_idx}.png"), 16777215)
        def depthimg2Meters(depth, cam):
            near = cam['cam_config']['nearval']
            far = cam['cam_config']['farval']
            image = near + depth * (far - near)
            return image
        depth_static = depthimg2Meters(depth_static, {"cam_config": static_cam_config})
        depth_gripper = depthimg2Meters(depth_gripper, {"cam_config": gripper_cam_config})
        static_extrinsic_matrix = frames[file_idx].misc['front_camera_extrinsics']
        gripper_extrinsic_matrix = frames[file_idx].misc['wrist_camera_extrinsics']
        static_extrinsic_matrix = np.linalg.inv(static_extrinsic_matrix) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 单位矩阵的逆和转置是一样的
        gripper_extrinsic_matrix = np.linalg.inv(gripper_extrinsic_matrix) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
        # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.0], #  [ 0.016149   -0.3464728   0.84676421  1.        ] [0.47925746 0.38417864 1.57344687 1.        ]
            [0, 1, 0, 0.0], # 
            [0, 0, 1, 0.7], # 
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [-1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {
            "rgb_static": rgb,
            "rgb_gripper": rgb2,
            "depth_static": depth_static,
            "depth_gripper": depth_gripper,
            "rel_actions": action,
            "robot_obs": np.zeros(20),
            "scene_obs": np.zeros(20),
            "calib": {  "rgb_static": { "extrinsic_matrix": static_extrinsic_matrix,
                                        "intrinsic_matrix": frames[file_idx].misc['front_camera_intrinsics'],
                                        "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                        "rgb_gripper":{ "extrinsic_matrix": gripper_extrinsic_matrix,
                                        "intrinsic_matrix": frames[file_idx].misc['wrist_camera_intrinsics'],
                                        "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}},
            "cam_config":{  "static": static_cam_config,
                            "gripper":gripper_cam_config},
            "state_matrix": gripper_extrinsic_matrix  # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # demo_name, end_idx, video_len = self.episode_lookup[idx]
        # end_idx+=1
        # episodes_dir = self.abs_datasets_dir / demo_name
        # with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
        #     frames = pickle.load(f)
        # start_idx = max(end_idx - window_size, 0)
        demo_name, start_idx, video_len = self.episode_lookup[idx]
        episodes_dir = self.abs_datasets_dir / demo_name
        with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
            frames = pickle.load(f)
        end_idx = start_idx + window_size

        episodes = [self._get_episode(file_idx, episodes_dir, frames) for file_idx in range(start_idx, end_idx)]

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[demo_name][-1]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        tasks=[x for x in "place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap".split(" ")]
        for task_str in tasks:
            if self.only_single_task and "open_drawer" not in task_str: continue # 测试使用
            task_path = abs_datasets_dir / task_str / "all_variations" / "episodes"
            for num_demo in os.listdir(task_path):
                task_name = f"{task_str}/all_variations/episodes/{num_demo}"
                episodes_dir = abs_datasets_dir / task_name
                with open(episodes_dir/"variation_descriptions.pkl", 'rb') as f:
                    lang_str = pickle.load(f)
                # if self.only_single_task and "3" not in lang_str[0]: continue # 测试使用
                lang_ann[task_name] = lang_str
                with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
                    obs = pickle.load(f)
                step_num = len(obs)
                for j in range(step_num - self.min_window_size): # 由于有+1操作，因此不加载最后一帧
                # for j in range(0+1, step_num): # YF：看calvin
                    episode_lookup.append((task_name, j, step_num))

        return episode_lookup, lang_ann


class DiskColosseumDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, episodes_dir, frames):
        
        def angle_between_angles(a, b):
            diff = b - a
            return (diff + np.pi) % (2 * np.pi) - np.pi
        def to_relative_action(actions, robot_obs, max_pos=0.03, max_orn=0.2):
            rel_pos = actions[:3] - robot_obs[:3]
            rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
            rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
            rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
            gripper = actions[-1:]
            return np.concatenate([rel_pos, rel_orn, gripper])
        def quate2euler(quat, gimble_fix=False):
            quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
            if quat[-1] < 0: quat = -quat
            euler = pybullet.getEulerFromQuaternion(quat)
            return euler
        r_euler = np.array(quate2euler(frames[file_idx].gripper_pose[3:7]))
        a_euler = np.array(quate2euler(frames[file_idx+1].gripper_pose[3:7]))
        if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8 and (r_euler[0]==0 or a_euler[0]==0 or r_euler[2]==0 or a_euler[2]==0): # 发生了万向节死锁
            assert False, 'Maybe gimbal lock has occurred.'
        robot_obs_euler = np.concatenate([frames[file_idx].gripper_pose[:3], r_euler, [1 if frames[file_idx].gripper_open>0.5 else -1]]) # YF: gripper_open=0/1, 注意要求的是-1和1
        actions_euler = np.concatenate([frames[file_idx+1].gripper_pose[:3], a_euler, [1 if frames[file_idx+1].gripper_open>0.5 else -1]])
        action = to_relative_action(actions_euler, robot_obs_euler[:6], max_pos=0.03, max_orn=0.2)
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        rgb = Image.open(episodes_dir/f"front_rgb/{file_idx}.png")
        rgb2 = Image.open(episodes_dir/f"wrist_rgb/{file_idx}.png")
        def calculate_fov(intrinsic_matrix, image_width):
            fx = intrinsic_matrix[0, 0]
            fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
            return fov
        with open(episodes_dir / "front_depth" / f"{file_idx}.pkl",'rb') as f:
            depth_static = pickle.load(f)
        with open(episodes_dir / "wrist_depth" / f"{file_idx}.pkl",'rb') as f:
            depth_gripper = pickle.load(f)
        static_extrinsic_matrix = frames[file_idx].misc['front_camera_extrinsics']
        gripper_extrinsic_matrix = frames[file_idx].misc['wrist_camera_extrinsics']
        static_extrinsic_matrix = np.linalg.inv(static_extrinsic_matrix) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 单位矩阵的逆和转置是一样的
        gripper_extrinsic_matrix = np.linalg.inv(gripper_extrinsic_matrix) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
        # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.0], #  [ 0.016149   -0.3464728   0.84676421  1.        ] [0.47925746 0.38417864 1.57344687 1.        ]
            [0, 1, 0, 0.0], # 
            [0, 0, 1, 0.7], # 
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [-1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {"rgb_static": rgb,
                "rgb_gripper": rgb2,
                "depth_static": depth_static,
                "depth_gripper": depth_gripper,
                "rel_actions": action,
                "robot_obs": np.zeros(20),
                "scene_obs": np.zeros(20),
                "calib":{   "rgb_static": { "extrinsic_matrix": static_extrinsic_matrix,
                                            "intrinsic_matrix": frames[file_idx].misc['front_camera_intrinsics'],
                                            "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                            "rgb_gripper":{ "extrinsic_matrix": gripper_extrinsic_matrix,
                                            "intrinsic_matrix": frames[file_idx].misc['wrist_camera_intrinsics'],
                                            "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}},
                "cam_config":{  "static": {"width": rgb.width, "height": rgb.height, "fov": calculate_fov(frames[file_idx].misc['front_camera_intrinsics'], rgb.height), "nearval": frames[file_idx].misc['front_camera_near'], "farval":  frames[file_idx].misc['front_camera_far']},
                                "gripper":{"width": rgb2.width, "height": rgb2.height, "fov": calculate_fov(frames[file_idx].misc['wrist_camera_intrinsics'], rgb2.height), "nearval": frames[file_idx].misc['wrist_camera_near'], "farval":  frames[file_idx].misc['wrist_camera_far']}},
                "state_matrix": gripper_extrinsic_matrix  # 注意此处是爪子矩阵
            }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # demo_name, end_idx, video_len = self.episode_lookup[idx]
        # end_idx+=1
        # episodes_dir = self.abs_datasets_dir / demo_name
        # with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
        #     frames = pickle.load(f)
        # start_idx = max(end_idx - window_size, 0)
        demo_name, start_idx, video_len = self.episode_lookup[idx]
        episodes_dir = self.abs_datasets_dir / demo_name
        with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
            frames = pickle.load(f)
        end_idx = start_idx + window_size
        # st=time.time()
        episodes = [self._get_episode(file_idx, episodes_dir, frames) for file_idx in range(start_idx, end_idx)]
        # print(f"loader: {time.time()-st}")
        episode = self._prepare_eposides(episodes)
        # print(f"loader: {time.time()-st}")
        if self.with_lang:
            episode["language"] = self.lang_ann[demo_name][-1]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        tasks = os.listdir(abs_datasets_dir)
        for task_str in tasks:
            if self.only_single_task and "open_drawer" not in task_str: continue # 测试使用
            task_path = abs_datasets_dir / task_str / "variation0" / "episodes"
            for num_demo in os.listdir(task_path):
                task_name = f"{task_str}/variation0/episodes/{num_demo}"
                episodes_dir = abs_datasets_dir / task_name
                with open(abs_datasets_dir/ task_str / "variation0" / "variation_descriptions.pkl", 'rb') as f:
                    lang_str = pickle.load(f)
                # if self.only_single_task and "3" not in lang_str[0]: continue # 测试使用
                lang_ann[task_name] = lang_str
                with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
                    obs = pickle.load(f)
                step_num = len(obs)
                for j in range(step_num - self.min_window_size): # 由于有+1操作，因此不加载最后一帧
                # for j in range(0+1, step_num): # YF：看calvin
                    episode_lookup.append((task_name, j, step_num))

        return episode_lookup, lang_ann


class DiskMTDataset(BaseMultiDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx, episodes_dir, episode_info) -> Path:
        def translate_intrinsic(left, top, intrinsic):
            intrinsic[0][2] -= left
            intrinsic[1][2] -= top
            return intrinsic
        def angle_between_angles(a, b):
            diff = b - a
            return (diff + np.pi) % (2 * np.pi) - np.pi
        def to_relative_action(actions, robot_obs, max_pos=0.03, max_orn=0.1):
            rel_pos = actions[:3] - robot_obs[:3]
            rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
            rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
            rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
            gripper = actions[-1:]
            return np.concatenate([rel_pos, rel_orn, gripper])
        r_euler = np.array(pybullet.getEulerFromQuaternion(episode_info['ee_pose_root'][file_idx, [4,5,6,3]]))
        a_euler = np.array(pybullet.getEulerFromQuaternion(episode_info['ee_pose_root'][file_idx+1, [4,5,6,3]]))
        if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8 and (r_euler[0]==0 or a_euler[0]==0 or r_euler[2]==0 or a_euler[2]==0): # 发生了万向节死锁
            assert False, 'Maybe gimbal lock has occurred.'
        robot_obs_euler = np.concatenate([episode_info['ee_pose_root'][file_idx,:3], r_euler, [1 if episode_info['gripper_target'][file_idx]>0.04 else -1]]) # YF: gripper_open=0/0.08, 注意要求的是-1和1
        actions_euler = np.concatenate([episode_info['ee_pose_root'][file_idx+1,:3], a_euler, [1 if episode_info['gripper_target'][file_idx+1]>0.04 else -1]])
        action = to_relative_action(actions_euler, robot_obs_euler[:6], max_pos=0.04, max_orn=0.12)
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]

        state = episode_info['ee_pose_root'][file_idx]
        arm_pose_mat = np.eye(4)
        arm_pose_mat[:3, 3] = state[:3]
        arm_pose_mat[:3, :3] = np.array(pybullet.getMatrixFromQuaternion(state[[4,5,6,3]])).reshape(3,3) # R.from_euler('xyz', state[3:6], False).as_matrix()

        static_camera_pose, gripper_camera_pose = episode_info['base_camera'].item()['pose'], episode_info['gripper_camera'].item()['pose']
        static_pose_mat = np.eye(4)
        static_pose_mat[:3, 3] = static_camera_pose[:3] #   - np.array([0.7, 0, 0.6])
        static_pose_mat[:3, :3] = np.array(pybullet.getMatrixFromQuaternion(static_camera_pose[[4,5,6,3]])).reshape(3,3) # R.from_euler('xyz',  R.from_quat(static_camera_pose[3:][[1, 2, 3, 0]]).as_euler('xyz', degrees=False), False).as_matrix()
        gripper_pose_mat = np.eye(4)
        gripper_pose_mat[:3, 3] = gripper_camera_pose[:3]
        gripper_pose_mat[:3, :3] = np.array(pybullet.getMatrixFromQuaternion(gripper_camera_pose[[4,5,6,3]])).reshape(3,3) # R.from_euler('xyz',  R.from_quat(gripper_camera_pose[3:][[1, 2, 3, 0]]).as_euler('xyz', degrees=False), False).as_matrix()
        calib = {'rgb_static': {'extrinsic_matrix': static_pose_mat,
                                'intrinsic_matrix': episode_info['base_camera'].item()['intrinsics'],
                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                'rgb_gripper': {'extrinsic_matrix': np.matmul(arm_pose_mat, gripper_pose_mat),
                                'intrinsic_matrix': episode_info['gripper_camera'].item()['intrinsics'],
                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}}

        rgb_static = Image.open(os.path.join(episodes_dir, "base_camera/rgb/%04d.png"%file_idx))
        rgb_gripper = Image.open(os.path.join(episodes_dir, "gripper_camera/rgb/%04d.png"%file_idx))
        crop_area = (80, 0, 560, 480)
        rgb_static = rgb_static.crop(crop_area)
        rgb_gripper = rgb_gripper.crop(crop_area)
        calib['rgb_static']['intrinsic_matrix'] = translate_intrinsic(80, 0, calib['rgb_static']['intrinsic_matrix'])
        calib['rgb_gripper']['intrinsic_matrix'] = translate_intrinsic(80, 0, calib['rgb_gripper']['intrinsic_matrix'])
        cam_config = {'static':{'height': rgb_static.height,
                                'width': rgb_static.width,
                                'fov': np.degrees(episode_info['base_camera'].item()['fovy'])}, #80
                     'gripper':{'height': rgb_gripper.height,
                                'width': rgb_gripper.width,
                                'fov': np.degrees(episode_info['gripper_camera'].item()['fovy'])}}

        calib['rgb_gripper']['extrinsic_matrix'] = np.linalg.inv(calib['rgb_gripper']['extrinsic_matrix']) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        calib['rgb_static']['extrinsic_matrix'] = np.linalg.inv(calib['rgb_static']['extrinsic_matrix']) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        R = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        calib['rgb_gripper']['extrinsic_matrix']=np.dot(calib['rgb_gripper']['extrinsic_matrix'], R)
        calib['rgb_static']['extrinsic_matrix'] = np.dot(calib['rgb_static']['extrinsic_matrix'], R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
            "rel_actions": action,
            "robot_obs": np.zeros(20),
            "scene_obs": np.zeros(20),
            "depth_static": np.array(Image.open(os.path.join(episodes_dir, "base_camera/depth/%04d.png"%file_idx)))[:, 80:-80]/1000,
            "depth_gripper": np.array(Image.open(os.path.join(episodes_dir, "gripper_camera/depth/%04d.png"%file_idx)))[:, 80:-80]/1000,
            "calib":calib,
            "cam_config": cam_config,
            "state_matrix": calib['rgb_gripper']['intrinsic_matrix'] # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        episode_id, start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        episodes_dir = os.path.join(self.abs_datasets_dir, "episode_%07d" % episode_id)
        episode_info = np.load(os.path.join(episodes_dir, 'episode_info.npz'),allow_pickle=True)
        episodes = [self._get_episode(file_idx, episodes_dir, episode_info) for file_idx in range(start_idx, end_idx)]

        episode = self._prepare_eposides(episodes)
        
        if self.with_lang:
            episode["language"] = self.lang_ann[episode_id]
            # if self.text_aug:
            #     task = self.lang_task[self.lang_lookup[idx]]
            #     enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
            #     episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()
        with open(os.path.join(abs_datasets_dir, 'data_info.json'), 'r') as f:
            info = json.load(f)
        
        episode_lookup, lang_ann = [], {}
        self._episode_idx = info['val' if self.validation else 'train' + '_eps']
        for idx in self._episode_idx:
            lang_ann[idx] = info['language_goal'][str(idx)]
            episode_length = info['ep_length'][str(idx)]
            for i in range(0, episode_length + 1 - self.min_window_size):
                episode_lookup.append((idx, i))
        return episode_lookup, lang_ann



class DiskChoresDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        use_static=False,
        nav_history_len=20,
        SS=0.95,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.use_static = use_static
        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)
        voxel_range = [[-2.0, 2.0], [-1.0, 3.0], [0.0, 1.0]]
        voxel_size = [0.0125*4, 0.0125*4, 0.0125*2]
        self.vfe_generator = OccupancyVFE(voxel_range, voxel_size) # 重新定义区域大小，导航需要更大的范围   
        self.nav_history_len = nav_history_len
        print(f"nav_history_len: {nav_history_len}")
        self.SS = SS
        print(f"SS: {SS}")

    def _read_history(self, house_traj_frame, nav_history_len=20, distance_threshold=0.1, SS=0.95):
        def _repeat_frame(matrix_a, matrix_b, threshold=0.1):
            """
            判断matrix_a与matrix_b中哪些帧的欧氏距离小于threshold，返回这些帧的索引
            """
            matrix_a = np.array(matrix_a)   # shape: (3,)
            matrix_b = np.array(matrix_b)   # shape: (l, 3)
            distances = np.linalg.norm(matrix_b - matrix_a, axis=1)
            repeat_indices = np.where(np.abs(distances) < threshold)[0]
            return repeat_indices
        
        def _similar_frame(vision_gripper_pool_np, vision_gripper_list, indices, SS):
            """
            判断vision_gripper_pool_np与vision_gripper_list中indices指定的帧是否有相似度大于SS的
            """
            if len(indices) == 0:
                return False
            vision_gripper_list = np.array(vision_gripper_list)
            selected = vision_gripper_list[indices]  # shape: (n_repeat, d) # navigation和eqa的这个地方维度不同，nav是(n_repeat, d)，eqa是(n_repeat, 1, d)
            # 判断selected的形状是否为3维，如果是则将前两个维度合并
            if len(selected.shape) == 3:
                selected = selected.reshape(-1, selected.shape[-1])

            v = vision_gripper_pool_np.reshape(1, -1)  # shape: (1, d)
            dot_products = np.dot(selected, v.T).flatten()
            norms_list = np.linalg.norm(selected, axis=1)
            norm_v = np.linalg.norm(vision_gripper_pool_np)
            cosine_similarities = dot_products / (norms_list * norm_v + 1e-8)
            return np.any(cosine_similarities > SS)

        house_id, traj_id, frame_id = house_traj_frame

        if frame_id < nav_history_len:
            history_list = [0] * (nav_history_len - frame_id - 1) + list(range(frame_id + 1))
        else:
            history_list = list(range(frame_id - nav_history_len + 1, frame_id + 1))
        history_list.reverse()

        replay_path = os.path.join(self.abs_datasets_dir, 'train')
        global_pose = None
        past_frame_id = None
        vision_gripper_list, pose_list = [], []
        for local_frame_id in history_list:
            output_dir = os.path.join(replay_path, str(house_id).zfill(6), str(traj_id))
            if not os.path.exists(output_dir):
                print(output_dir, "is not exists!")
            prefix = "_static" if self.use_static else ""
            extra_feat = "extra_feat" if self.use_static else "vision_gripper"
            output_hdf5 = os.path.join(output_dir, f"vit{prefix}_feat_{str(local_frame_id)}.hdf5")
            try:
                with h5py.File(output_hdf5, "r") as f_out:
                    robot_matrix_np = f_out["robot_matrix"][:] # np, (4, 4)
                    vision_gripper_pool_np = f_out[extra_feat][:] # np, (1024,) navigation和eqa的这个地方维度不同，nav是(1024,)，eqa是(1, 1024)
                    pose = robot_matrix_np[:3, 3] # np, (3,)
            except Exception as e:
                # with open(os.path.join(replay_path, "error_log.txt"), "a") as log_file:
                #     log_file.write(f"{output_hdf5}\n")
                return None, None
                
            if local_frame_id == frame_id: # 第一帧原点不用计算，直接保存。当frame_id=0的时候一直在这
                global_pose = pose
                local_pose = pose - global_pose
                vision_gripper_list.append(vision_gripper_pool_np)
                pose_list.append(local_pose)
            else:
                if past_frame_id == 0 and local_frame_id == 0: # 当剩下的都是0，直接保存
                    vision_gripper_list.append(vision_gripper_list[-1])
                    pose_list.append(pose_list[-1])
                else:
                    local_pose = pose - global_pose

                    repeat_indices = _repeat_frame(local_pose, pose_list, threshold=distance_threshold)
                    if not _similar_frame(vision_gripper_pool_np, vision_gripper_list, repeat_indices, SS):
                        vision_gripper_list.append(vision_gripper_pool_np)
                        pose_list.append(local_pose)
                    else:
                        new_frame = 0 if history_list[-1] == 0 else history_list[-1] - 1
                        history_list.append(new_frame)

            past_frame_id = local_frame_id

            if len(vision_gripper_list) == nav_history_len and len(pose_list) == nav_history_len:
                break

        vision_gripper_np = np.array(vision_gripper_list) # nav_history_len 1024
        pose_np = np.array(pose_list) # nav_history_len 3

        return vision_gripper_np, pose_np

    def _get_episode(self, start_idx, end_idx, traj, img_dir, task, episode_id):
        action2idx = {  'm': 0,     'r': 1,     'l': 2,     'b': 3,     'end': 4, 'sub_done': 5,    'ls': 6,    'rs': 7, \
                        'p': 8,     'zm': 9,    'zp': 10,   'yp': 11,   'ym': 12, 'wp': 13,         'wm': 14, \
                        'yms': 15,  'zms': 16,  'zps': 17,  'yps': 18,  'd': 19}
        flag = [] 
        # action.shape (13, 7)
        action = np.stack([[0]*6+[action2idx[traj["actions"][file_idx].decode('utf-8')]] for file_idx in range(start_idx, end_idx)]) # 注意假如后边与坐标对齐的时候，需要根据底下R矩阵来对应修改aciton

        crop_area = (86, 0, 310, 224)
        rgb_statics, rgb_grippers, depth_statics, depth_grippers = [], [], [], []
        his_vision, his_pose = [], []
        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_idx in range(start_idx, end_idx): 
            rgb_static = Image.open(os.path.join(img_dir, f'static_rgb_{file_idx}.jpg')) # 224 396 3
            rgb_gripper = Image.open(os.path.join(img_dir, f'gripper_rgb_{file_idx}.jpg')) # 224 396 3
            depth_static = np.load(os.path.join(img_dir, f"static_depth.npz"))['arr_0'][file_idx] # 224 396
            depth_gripper = np.load(os.path.join(img_dir, f"gripper_depth.npz"))['arr_0'][file_idx] # 224 396
            rgb_static = rgb_static.crop(crop_area)
            rgb_gripper = rgb_gripper.crop(crop_area)
            depth_static = depth_static[:, 86:-86]
            depth_gripper = depth_gripper[:, 86:-86]
            rgb_statics.append(rgb_static)
            rgb_grippers.append(rgb_gripper)
            depth_statics.append(depth_static)
            depth_grippers.append(depth_gripper)
            house_traj_frame = [int(task), int(episode_id), int(file_idx)]
            flag.append(house_traj_frame) # 房间号，轨迹号，帧号
            vision_gripper_np, pose_np = self._read_history(house_traj_frame, nav_history_len=self.nav_history_len, SS=self.SS) # nav_history_len 1024, nav_history_len 3
            his_vision.append(vision_gripper_np) # t nav_history_len 1024
            his_pose.append(pose_np) # t nav_history_len 3

        base_pose = traj["base_matrix"][start_idx:end_idx][()] # 12 4 4
        static_extrinsic_matrix = traj["calib/rgb_static/extrinsic_matrix"][start_idx:end_idx][()] # 12 4 4
        gripper_extrinsic_matrix = traj["calib/rgb_gripper/extrinsic_matrix"][start_idx:end_idx][()] # 12 4 4
        static_extrinsic_matrix = np.linalg.inv(static_extrinsic_matrix) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])   #注意这里没有求逆
        gripper_extrinsic_matrix = np.linalg.inv(gripper_extrinsic_matrix) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界
        static_extrinsic_matrix = np.einsum('mij,mjk->mik', static_extrinsic_matrix, base_pose)
        gripper_extrinsic_matrix = np.einsum('mij,mjk->mik', gripper_extrinsic_matrix, base_pose)
        # # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.0],    # 
            [0, 1, 0, 1.0],     # 
            [0, 0, 1, 0.0],    #  
            [0, 0, 0, 1] 
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        base_pose = np.einsum('mij,jk->mik', base_pose, T_translate)
        R = np.array([
            [1, 0, 0, 0],
            [0,  0, -1, 0],
            [0,  1, 0, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, R) # 往右X轴；往下为Y轴；往前为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        base_pose = np.einsum('mij,jk->mik', base_pose, R) # 往右X轴；往下为Y轴；往前为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        static_intrinsic_matrix = traj["calib/rgb_static/intrinsic_matrix"][start_idx:end_idx][()]
        gripper_intrinsic_matrix = traj["calib/rgb_gripper/intrinsic_matrix"][start_idx:end_idx][()]
        static_intrinsic_matrix[:, 0, 0] = static_intrinsic_matrix[:, 1, 1]
        static_intrinsic_matrix[:, 0, 2] = static_intrinsic_matrix[:, 1, 2]
        gripper_intrinsic_matrix[:, 0, 0] = gripper_intrinsic_matrix[:, 1, 1]
        gripper_intrinsic_matrix[:, 0, 2] = gripper_intrinsic_matrix[:, 1, 2]

        cam_config = {  "static": {"height": rgb_statics[0].height, "width": rgb_statics[0].width, "fov": traj["cam_config/static_fov"][start_idx][()]},
                            "gripper":{"height": rgb_grippers[0].height, "width": rgb_grippers[0].width, "fov": traj["cam_config/gripper_fov"][start_idx][()]},}
        return [{   "rgb_static": rgb_static_,
                    "rgb_gripper": rgb_gripper_,
                    "depth_static": depth_static_,
                    "depth_gripper": depth_gripper_,
                    "rel_actions": action_,
                    "robot_obs": np.zeros(20),
                    "scene_obs": np.zeros(20),
                    "calib": {  'rgb_static': { 'extrinsic_matrix': static_extrinsic_matrix_,         
                                                'intrinsic_matrix': static_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                                'rgb_gripper': {'extrinsic_matrix': gripper_extrinsic_matrix_,
                                                'intrinsic_matrix': gripper_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                            },
                    "cam_config": cam_config,
                    "state_matrix": gripper_extrinsic_matrix_,  # 注意此处是爪子矩阵
                    "his_vision": his_vision_, # nav_history_len 1024
                    "his_pose": his_pose_, # nav_history_len 3
        }for (rgb_static_, rgb_gripper_, depth_static_, depth_gripper_, action_, \
            static_extrinsic_matrix_, static_intrinsic_matrix_, gripper_extrinsic_matrix_, gripper_intrinsic_matrix_, his_vision_, his_pose_) \
        in zip(rgb_statics, rgb_grippers, depth_statics, depth_grippers, action, \
            static_extrinsic_matrix, static_intrinsic_matrix, gripper_extrinsic_matrix, gripper_intrinsic_matrix, his_vision, his_pose)
        ]
坐标系到本体上。

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx=random.choice(list(range(12))); print("idx==0")
        dataset_file, episode_id, start_idx, episode_length, task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        with h5py.File(dataset_file, "r") as f_out:
            house = f_out[str(task)]
            traj = house[str(episode_id)]
            # zip_file = os.path.join(self.abs_datasets_dir, "train", str(task), f"{str(episode_id)}.zip")
            img_dir = os.path.join(self.abs_datasets_dir, "train", str(task), f"{str(episode_id)}")
            episodes = self._get_episode(start_idx, end_idx, traj, img_dir, task, episode_id)

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[f"{task}_{episode_id}"]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        subset = "train" # zyf 默认为训练集！！！
        house_id_to_sub_house_id_json = os.path.join(abs_datasets_dir, f"house_id_to_sub_house_id_{subset}.json")
        with open(house_id_to_sub_house_id_json, "r") as f:
            house_id_to_sub_house_id = json.load(f)
        house_id_to_sub_house_id = dict(sorted(house_id_to_sub_house_id.items())) # zyf 一定要排序，否则会遇到还未处理的house_id而无法读取hdf5文件！！！
        house_len = len(house_id_to_sub_house_id)
        if self.only_single_task: house_len = 3 # 测试使用
        for task in list(house_id_to_sub_house_id.keys())[:house_len]: # 加载不同房间
            dataset_file = os.path.join(abs_datasets_dir, subset, task, "hdf5_sensors.hdf5")

            try:
                f_out = h5py.File(dataset_file, "r")
            except Exception as e:
                print(f"Error: {e}")
                continue
        
            house = f_out[str(task)]
            for traj_id in house_id_to_sub_house_id[task]: # 记录每一条轨迹的信息
                # 构建episode_lookup
                dir_path = os.path.join(abs_datasets_dir, subset, str(task), str(traj_id)) # zyf测试
                if not os.path.exists(dir_path): continue
                if traj_id not in house: continue
                traj = house[str(traj_id)]
                if len(traj.keys()) == 0: continue

                lang_ann[f"{task}_{traj_id}"] = traj["language"][()].decode('utf-8')
                episode_length = traj["base_matrix"].shape[0]
                for start_frame_index in range(0, episode_length + 1 - self.min_window_size):
                    episode_lookup.append((dataset_file, traj_id, start_frame_index, episode_length, task))

            f_out.close()

        return episode_lookup, lang_ann


class DiskChoresVQADataset(DiskChoresDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert self.use_static == True, "In vqa dataset, self.use_static must be True!!!"
        lang_ann, episode_lookup = {}, []
        subset = "train" # zyf 默认为训练集！！！
        house_id_to_sub_house_id_json = os.path.join(abs_datasets_dir, f"house_id_to_sub_house_id_{subset}.json")
        with open(house_id_to_sub_house_id_json, "r") as f:
            house_id_to_sub_house_id = json.load(f)
        house_id_to_sub_house_id = dict(sorted(house_id_to_sub_house_id.items())) # zyf 一定要排序，否则会遇到还未处理的house_id而无法读取hdf5文件！！！
        house_len = len(house_id_to_sub_house_id)

        cur_save_path = '~/zhongyufeng/spoc/fifteen/ObjectNavType/replay/train_vqa_all.jsonl'
        vqa_info = {}
        with open(cur_save_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每一行的JSON数据
                info = json.loads(line)
                key = list(info.keys())[0]
                vqa_info[key] = info[key]

        if self.only_single_task: house_len = 3 # 测试使用
        for task in list(house_id_to_sub_house_id.keys())[:house_len]: # 加载不同房间
            dataset_file = os.path.join(abs_datasets_dir, subset, task, "hdf5_sensors.hdf5")

            try:
                f_out = h5py.File(dataset_file, "r")
            except Exception as e:
                print(f"Error: {e}")
                continue
        
            house = f_out[str(task)]
            for traj_id in house_id_to_sub_house_id[task]: # 记录每一条轨迹的信息
                # 构建episode_lookup
                dir_path = os.path.join(abs_datasets_dir, subset, str(task), str(traj_id)) # zyf测试
                if not os.path.exists(dir_path): continue
                if traj_id not in house: continue
                traj = house[str(traj_id)]
                if len(traj.keys()) == 0: continue
                
                cur_info = vqa_info[task][traj_id]
                lang_ann[f"{task}_{traj_id}"] = f"Question: {cur_info['question']}. Answer: {cur_info['answer']}"

                episode_length = traj["base_matrix"].shape[0]
                for start_frame_index in range(episode_length - 1, episode_length + 1 - self.min_window_size):
                    episode_lookup.append((dataset_file, traj_id, start_frame_index, episode_length, task))

            f_out.close()

        return episode_lookup, lang_ann


class DiskChoresRoomVQADataset(DiskChoresDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert self.use_static == True, "In vqa dataset, self.use_static must be True!!!"
        lang_ann, episode_lookup = {}, []
        subset = "train" # zyf 默认为训练集！！！
        house_id_to_sub_house_id_json = os.path.join(abs_datasets_dir, f"house_id_to_sub_house_id_{subset}.json")
        with open(house_id_to_sub_house_id_json, "r") as f:
            house_id_to_sub_house_id = json.load(f)
        house_id_to_sub_house_id = dict(sorted(house_id_to_sub_house_id.items())) # zyf 一定要排序，否则会遇到还未处理的house_id而无法读取hdf5文件！！！
        house_len = len(house_id_to_sub_house_id)

        cur_save_path = '~/zhongyufeng/spoc/fifteen/ObjectNavRoom/replay/train_vqa_new_1739006908.4866471.jsonl'
        vqa_info = {}
        with open(cur_save_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每一行的JSON数据
                info = json.loads(line)
                key = list(info.keys())[0]
                vqa_info[key] = info[key]

        if self.only_single_task: house_len = 3 # 测试使用
        lang_ann[f"Question"] = f"Question: {vqa_info['question']} "
        for task in list(house_id_to_sub_house_id.keys())[:house_len]: # 加载不同房间
            dataset_file = os.path.join(abs_datasets_dir, subset, task, "hdf5_sensors.hdf5")

            try:
                f_out = h5py.File(dataset_file, "r")
            except Exception as e:
                print(f"Error: {e}")
                continue
        
            house = f_out[str(task)]
            for traj_id in house_id_to_sub_house_id[task]: # 记录每一条轨迹的信息
                # 构建episode_lookup
                dir_path = os.path.join(abs_datasets_dir, subset, str(task), str(traj_id)) # zyf测试
                if not os.path.exists(dir_path): continue
                if traj_id not in house: continue
                traj = house[str(traj_id)]
                if len(traj.keys()) == 0: continue

                episode_length = traj["base_matrix"].shape[0]
                for start_frame_index in range(0, episode_length + 1 - self.min_window_size):
                    episode_lookup.append((dataset_file, traj_id, start_frame_index, episode_length, task))
                    cur_info = vqa_info[task][traj_id][str(start_frame_index)]
                    lang_ann[f"{task}_{traj_id}_{start_frame_index}"] = f"Answer: {cur_info['answer']}"

            f_out.close()

        return episode_lookup, lang_ann

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx=random.choice(list(range(12))); print("idx==0")
        dataset_file, episode_id, start_idx, episode_length, task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        with h5py.File(dataset_file, "r") as f_out:
            house = f_out[str(task)]
            traj = house[str(episode_id)]
            # zip_file = os.path.join(self.abs_datasets_dir, "train", str(task), f"{str(episode_id)}.zip")
            img_dir = os.path.join(self.abs_datasets_dir, "train", str(task), f"{str(episode_id)}")
            episodes = self._get_episode(start_idx, end_idx, traj, img_dir, task, episode_id)

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[f"Question"] + self.lang_ann[f"{task}_{episode_id}_{start_idx}"]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode


class DiskMultiDataset(Dataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        data_types,
        datasets_dir,
        **kwargs: Any,
    ):
        # kwargs_cpy = copy.deepcopy(kwargs)
        data_path_list = kwargs.pop("data_path_list")
        state_matrixs_path = kwargs.pop("state_matrixs_path")
        self.datas_loder = []
        self.datas_len = [0] # 注意datas_len的长度比self.datas_loder多一个
        if not isinstance(data_types, list):
            data_types = [data_types] 
            datasets_dir = [datasets_dir]
            state_matrixs_path = [state_matrixs_path]
            data_path_list = [data_path_list]
        for ith, (data_type, dataset_dir) in enumerate(zip(data_types, datasets_dir)):
            
            if data_type == "calvin":
                data_loder = DiskCalvinDataset(
                    datasets_dir=Path(os.path.join(dataset_dir, "training")),
                    state_matrixs_path=state_matrixs_path[ith],
                    data_path_list=data_path_list[ith],
                    **kwargs)
            elif 'metaworld' in data_type:
                data_loder=DiskMetaWorldDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'libero' in data_type:
                data_loder=DiskLiberoDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'robocasa' in data_type:
                data_loder=DiskRobocasaDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'robomimic' in data_type:
                data_loder=DiskRoboMimicDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'maniskill2' in data_type:
                data_loder=DiskManiSkill2Dataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'rlbench' in data_type:
                data_loder=DiskRlbenchDataset(
                    datasets_dir=Path(os.path.join(dataset_dir)), # train
                    **kwargs)
            elif 'colosseum' in data_type:
                data_loder=DiskColosseumDataset(
                    datasets_dir=Path(os.path.join(dataset_dir)), # train
                    **kwargs)
            elif 'mt' in data_type:
                data_loder=DiskMTDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'spocvqa' in data_type:
                data_loder=DiskChoresVQADataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'roomvqa' in data_type:
                data_loder=DiskChoresRoomVQADataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'chores' in data_type:
                data_loder=DiskChoresDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            else:
                assert False
            data_loder.data_name = data_type
            self.datas_loder.append(data_loder)
            self.datas_len.append(len(data_loder))
            print(f"loader {data_type} from {dataset_dir} with {len(data_loder)}")

        self.datas_len = np.cumsum(np.array(self.datas_len)) # 注意datas_len的长度比self.datas_loder多一个

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return self.datas_len[-1]

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        assert isinstance(idx, int)
        for i in range(len(self.datas_loder)):
            if idx >= self.datas_len[i] and idx <= self.datas_len[i+1]:
                # st = time.time()
                sample = self.datas_loder[i].__getitem__(idx-self.datas_len[i])
                # if torch.distributed.get_rank() == 0:
                #     # self.datas_loder[i].tmp.append(time.time() - st)
                #     print(self.datas_loder[i].data_name, idx-self.datas_len[i], time.time() - st)
                return sample

    def collater(self, sample):
        # st = time.time() 这里的sample是bs个__getitem__获得的sample（sequence）
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample])) # 4 13 7
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        static_extrinsic_matrix = torch.stack([s["calib_obs"]["static_extrinsic_matrix"] for s in sample])
        static_intrinsic_matrix = torch.stack([s["calib_obs"]["static_intrinsic_matrix"] for s in sample])
        static_distCoeffs_matrix = torch.stack([s["calib_obs"]["static_distCoeffs_matrix"] for s in sample])
        gripper_extrinsic_matrix = torch.stack([s["calib_obs"]["gripper_extrinsic_matrix"] for s in sample])
        gripper_intrinsic_matrix = torch.stack([s["calib_obs"]["gripper_intrinsic_matrix"] for s in sample])
        gripper_distCoeffs_matrix = torch.stack([s["calib_obs"]["gripper_distCoeffs_matrix"] for s in sample])
        state_matrix = torch.stack([s["calib_obs"]["state_matrix"] for s in sample])
        static_fovs = torch.stack([s["calib_obs"]["static_fov"] for s in sample])
        gripper_fovs = torch.stack([s["calib_obs"]["gripper_fov"] for s in sample])
        his_vision = torch.stack([s["his_obs"]["his_vision"] for s in sample]) # 4 12 20 1024
        his_pose = torch.stack([s["his_obs"]["his_pose"] for s in sample]) # 4 12 20 3

        # 使用多线程并行处理样本
        def process_sample(flag, sample, datas_loder):
            # st = time.time()
            if flag == 'text':
                stacked_language = [s["lang"] for s in sample]
                text_tensors, attention_mask = datas_loder[0].text_fn(stacked_language)
                # print(f"YF: {flag} {time.time()-st}")
                return flag, text_tensors, attention_mask
            elif flag == 'occ':
                pcd = torch.stack([s["pcd_obs"]["pcd"] for s in sample])
                # pcd = [s["pcd_obs"]["pcd"] for s in sample]
                # print(f"YF: {flag} {time.time()-st}")
                return flag, pcd
            elif flag == 'static':
                # image_tensors = torch.stack([datas_loder[0].image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
                image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
                if datas_loder[0].rgb_pad != -1:
                    bs, seq_len = image_tensors.shape[:2]
                    if datas_loder[0].traj_cons:
                        image_tensors = datas_loder[0].rgb_shift.forward_traj(image_tensors)
                    else:
                        image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                        image_tensors = datas_loder[0].rgb_shift(image_tensors)
                        image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
                # print(f"YF: {flag} {time.time()-st}")
                return flag, image_tensors
            elif flag == 'gripper':
                # gripper_tensors = torch.stack([datas_loder[0].image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
                gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])
                if datas_loder[0].gripper_pad != -1:
                    bs, seq_len = gripper_tensors.shape[:2]
                    if datas_loder[0].traj_cons:
                        gripper_tensors = datas_loder[0].gripper_shift.forward_traj(gripper_tensors)
                    else:
                        gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                        gripper_tensors = datas_loder[0].gripper_shift(gripper_tensors)
                        gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
                # print(f"YF: {flag} {time.time()-st}")
                return flag, gripper_tensors
        partial_process_sample = functools.partial(process_sample, sample=sample, datas_loder=self.datas_loder)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(partial_process_sample, ['text', 'occ', 'static', 'gripper']))
        result_dict = {flag: result for flag, *result in results}
        # print(f"YF: collater2: {time.time()-st}")
        text_tensors, attention_mask = result_dict['text'] # 4 146; 4 146
        pcd = result_dict['occ'][0] # 4 13 80 80 40 4
        image_tensors = result_dict['static'][0] # 4 13 3 224 224
        gripper_tensors = result_dict['gripper'][0] # 4 13 3 224 224
        # print(f"YF: collater3: {time.time()-st}")
        robot_obs = torch.zeros(1)
        if self.datas_loder[0].act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.datas_loder[0].window_size, self.datas_loder[0].act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.datas_loder[0].window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.datas_loder[0].act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.datas_loder[0].window_size, self.datas_loder[0].act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.datas_loder[0].window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.datas_loder[0].act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.datas_loder[0].act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.datas_loder[0].act_step-1)]
            state_tensors = state_tensors[:, :-(self.datas_loder[0].act_step-1)]
        
        idxes = torch.stack([torch.from_numpy(np.array(s["idx"])) for s in sample]) # 4
        # print(f"YF: collater: {time.time()-st}")
        return image_tensors, (text_tensors, attention_mask), action_tensors, gripper_tensors, state_tensors, robot_obs,(static_extrinsic_matrix,
                static_intrinsic_matrix,static_distCoeffs_matrix,gripper_extrinsic_matrix,gripper_intrinsic_matrix,gripper_distCoeffs_matrix,state_matrix,static_fovs,gripper_fovs),pcd,idxes,his_vision,his_pose


def CalvinEvalSeq(
                 env,
                 dataset_path,
                 initial_state, eval_sequence,
                 val_annotations,
                 task_oracle,
                 transforms={},
                 EP_LEN = 360
                 ):
        if env is None:
            env = get_env(Path(dataset_path), show_gui=False)  # make_env(dataset_path)
        """
        Evaluates a sequence of language instructions.
        """
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        reset = False
        success_counter = 0
        is_reset = False
        for subtask_i, subtask in enumerate(eval_sequence):

            if subtask_i > 0: break  # 只测试task 0

            planned_actions = []
            if robot_obs is not None and scene_obs is not None and reset:
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                is_reset = True
            obs = env.get_obs()
            # get lang annotation for subtask
            lang_annotation = val_annotations[subtask][0]
            lang_annotation = lang_annotation.split('\n')[0]
            if '\u2019' in lang_annotation:
                lang_annotation.replace('\u2019', '\'')
            start_info = env.get_info()

            success = False # determine success or not

            for step in range(EP_LEN):
                
                ret = {
                    "success_before": success,
                    "lang": lang_annotation,
                    "subtask_i": subtask_i,
                    "eval_sequence": eval_sequence,
                    "success_counter": success_counter,
                    "step_cur": step,
                    "step_max": EP_LEN,
                    "is_reset": is_reset,
                    "rgb_static": transforms['rgb_static'](Image.fromarray(obs['rgb_obs']['rgb_static'])),
                    "rgb_gripper": transforms['rgb_gripper'](Image.fromarray(obs['rgb_obs']['rgb_gripper'])),
                    "rgb_static_ori": obs['rgb_obs']['rgb_static'],
                    "rgb_gripper_ori": obs['rgb_obs']['rgb_gripper'],
                    "robot_obs": obs['robot_obs'],
                    "done": False
                }
                action = yield ret
                if len(planned_actions) == 0:
                    if action.shape == (7,):
                        planned_actions.append(action)
                    else:
                        planned_actions.extend([action[i] for i in range(action.shape[0])])
                action = planned_actions.pop(0)
                obs, _, _, current_info = env.step(action)

                # check if current step solves a task
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    success = True
                    break
            if not success:
                break
            else:
                success_counter += 1
        
        ret = {"eval_sequence": eval_sequence,
                "success_counter": success_counter,
                "done": True
                }
        _ = yield ret
        
class CalvinSim(Dataset):
    def __init__(self,  
                 dataset_path,
                 calvin_conf_path,
                 calvin_seq_path,
                 transforms={},
                 NUM_SEQUENCES = 300,
                 ):
        super(CalvinSim, self).__init__()

        self.dataset_path = dataset_path

        conf_dir = Path(calvin_conf_path) # 
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self.task_oracle = hydra.utils.instantiate(task_cfg)

        self.val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml") # 导入语言标注
        with open(calvin_seq_path, 'r') as f:
            self.eval_sequences = json.load(f)
        self.eval_sequences = self.eval_sequences[:NUM_SEQUENCES]
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.eval_sequences)
    
    def __getitem__(self, idx: int, fixed_seed=False):

        initial_state, eval_sequence = self.eval_sequences[idx]
        EP_LEN = 180
        return {"generator": CalvinEvalSeq,
                "dataset_path": self.dataset_path,
                "initial_state": initial_state,
                "eval_sequence": eval_sequence,
                "val_annotations": self.val_annotations,
                "task_oracle": self.task_oracle,
                "transforms": self.transforms,
                "EP_LEN": EP_LEN
                }
    
    def collater(self, sample):

        return sample

class CalvinDataset(Dataset):
    """Naive implementation of dataset to store
    calvin debug dataset, may be changed to WDS for the full dataset
    """

    def __init__(self, image_fn, text_fn, dataset_path, is_train=True) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.image_fn = image_fn
        self.text_fn = text_fn

        tag = "training" if is_train else "validation"
        self.file_prefix = f"{self.dataset_path}/{tag}"
        self.anns = np.load(
            f"{self.file_prefix}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
        ).item()
        self.tag = tag

    def __len__(self):
        return len(self.anns["info"]["indx"])

    def __getitem__(self, index):
        task = self.anns["language"]["task"][index]
        text = self.anns["language"]["ann"][index]
        st, ed = self.anns["info"]["indx"][index]
        # CJ: randomly sample a datapoint in the episode
        frame = random.randint(st, ed)
        frame = np.load(
            f"{self.file_prefix}/episode_{frame:07d}.npz"
        )  # , allow_pickle=True (lazy load)
        rgb_static = Image.fromarray(frame["rgb_static"])
        rgb_gripper = Image.fromarray(frame["rgb_gripper"])
        actions = np.array(frame["rel_actions"])
        
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        return rgb_static, text, actions

    def collater(self, sample):
        images = [s[0] for s in sample]
        texts = [s[1] for s in sample]
        actions = [s[2] for s in sample]

        image_tensors = self.image_fn(images)
        text_tensors = self.text_fn(texts)
        action_tensors = torch.FloatTensor(np.stack(actions))
        return image_tensors, text_tensors, action_tensors


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix(), allow_pickle=True)
    #return np.load(filename.as_posix())


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image


def preprocess_text_calvin(sample, tokenizer, sample_mode=-1, window_size=12):
    tokenizer.padding_side = "right"
    # sample = [
    #     # (f"{s.strip()}{tokenizer.eos_token}")
    #     # for s in sample
    #     (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    # ]
    sample_modes = [0,1,2,3,4]
        #sample_modes = [3]
    sample_mode = random.choice(sample_modes) if sample_mode<0 else sample_mode 
    # YF：下边添加f,|endofchunk|>{tokenizer.eos_token}变为<|endofchunk|><|endoftext|>
    if sample_mode == 0: # 图片+action 
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<static_s><static0><static1><static2><static3><static4><static5><static6><static7><static_e><gripper_s><gripper0><gripper1><gripper2><gripper3><gripper4><gripper5><gripper6><gripper7><gripper_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 1: # action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 2: # occ+action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<obs_s><obs0><obs1><obs2><obs3><obs4><obs5><obs6><obs7><obs_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 3: # image+occ+action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<static_s><static0><static1><static2><static3><static4><static5><static6><static7><static_e><gripper_s><gripper0><gripper1><gripper2><gripper3><gripper4><gripper5><gripper6><gripper7><gripper_e><obs_s><obs0><obs1><obs2><obs3><obs4><obs5><obs6><obs7><obs_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 4:  # occ+image+action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<obs_s><obs0><obs1><obs2><obs3><obs4><obs5><obs6><obs7><obs_e><static_s><static0><static1><static2><static3><static4><static5><static6><static7><static_e><gripper_s><gripper0><gripper1><gripper2><gripper3><gripper4><gripper5><gripper6><gripper7><gripper_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 5:  # vqa
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]

    text = tokenizer(
        sample,
        max_length=2048, # 1024,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"] # YF: input_ids为：50278, 29397,   598，其中50278为<image>， 29397为pick/slide/grasp等动词


def preprocess_interleaved(sample, tokenizer, clip_processor, sim_threshold):
    assert False # 可能有内存泄漏io.BytesIO
    info = json.loads(sample[0])
    tar_file_obj = io.BytesIO(sample[1])
    image_tar = tarfile.open(fileobj=tar_file_obj)
    sentences = info["text_list"]

    images, image_idxs = [], []
    for image_path, sim in zip(info["image_info"], info["similarity_matrix"]):
        # pick one image per sentence
        if info["image_info"][image_path]["matched_text_index"] in image_idxs:
            continue
        rawbytes = image_tar.extractfile(
            os.path.join(image_tar.getnames()[0], image_path)
        ).read()

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue
        if sim[info["image_info"][image_path]["matched_text_index"]] < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        image_idxs.append(info["image_info"][image_path]["matched_text_index"])

    if len(images) == 0:
        raise ValueError("No images in sample")

    # filter out images that are exact duplicates
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), MAX_NUM_IMAGES))
    images_tensors = images_tensors[keep_ixs]
    image_idxs = [image_idxs[ix] for ix in keep_ixs]

    # pad to 5 images
    if len(images_tensors) < MAX_NUM_IMAGES:
        zero_padding = torch.zeros(
            (MAX_NUM_IMAGES - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # add in <image> and <eoc> tokens
    # eoc after sentence = "sentence loss"
    for ix in image_idxs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"

    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )

    if num_images == 0:
        raise ValueError("No images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def get_coco_dataset(args, image_processor, tokenizer, epoch=0):
    assert False # YF 代码未用
    coco_data_dir = "path/to/coco/train2014"
    coco_ann = "path/to/coco/annotations/captions_train2014.json"
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    coco_dataset = CaptionDataset(coco_data_dir, coco_ann, preprocess_text_fn, image_processor)
    
    sampler = DistributedSampler(
        coco_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    
    dataloader = DataLoader(
        coco_dataset,
        batch_size=args.batch_size_vl,
        pin_memory=False,
        num_workers=args.workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=coco_dataset.collator,
        drop_last=True
    )
    
    return dataloader


def get_vqa_dataset(args, image_processor, tokenizer, epoch=0):
    assert False # YF 代码未用
    vqa_data_dir = "path/to/vqav2/train2014"
    vqa_questions = "path/to/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
    vqa_ann = "path/to/vqav2/v2_mscoco_train2014_annotations.json"
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    vqa_dataset = VQADataset(vqa_data_dir, vqa_questions, vqa_ann, preprocess_text_fn, image_processor)
    
    sampler = DistributedSampler(
        vqa_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    
    dataloader = DataLoader(
        vqa_dataset,
        batch_size=args.batch_size_vl,
        pin_memory=False,
        num_workers=args.workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=vqa_dataset.collator,
        drop_last=True
    )
    
    return dataloader


class DistributedSamplerWithSkip(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 跳过前几个样本
        self.num_skip_iters = 0

    def __iter__(self):
        it = super().__iter__()
        for _ in range(self.num_skip_iters):
            next(it)
        # 仅第一次调用时跳过样本，后续epoch不跳过
        self.num_skip_iters = 0
        return it
    
    def set_num_skip_iters(self, num_skip_iters):
        self.num_skip_iters = num_skip_iters


def get_multi_dataset(args, image_processor, tokenizer, dataset_type, epoch=0, floor=False):
    dataset_path = args.calvin_dataset

    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    # preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_image_fn = functools.partial(lambda X: X)
    transforms = dict()
    transforms["rgb_static"] = image_processor
    transforms["rgb_gripper"] = image_processor
    data_types = dataset_type if dataset_type in [["spocvqa"], ["roomvqa"]] else args.data_type
    sample_mode = args.vqa_sample_mode if dataset_type in [["spocvqa"], ["roomvqa"]] else args.sample_mode
    window_size = args.vqa_window_size if dataset_type in [["spocvqa"], ["roomvqa"]] else args.window_size
    batch_size = args.vqa_batch_size if dataset_type in [["spocvqa"], ["roomvqa"]] else args.batch_size_calvin
    use_static = args.vqa_use_static if dataset_type in [["spocvqa"], ["roomvqa"]] else args.use_static
    workers = args.vqa_workers if dataset_type in [["spocvqa"], ["roomvqa"]] else args.workers
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer, sample_mode=sample_mode, window_size=window_size)
    SS = args.SS
    
    print(f"********************YF:sample_mode={sample_mode}***********************")
    print(f"data_types: {data_types}")
    print(f"window_size: {window_size}")
    print(f"batch_size: {batch_size}")
    print(f"use_static: {use_static}")
    print(f"workers: {workers}")
    print(f"SS: {SS}")
    if hasattr(args, 'data_tasks_groups'):
        if args.data_tasks_groups == 'None':  data_tasks_groups = None
        else: data_tasks_groups = args.data_tasks_groups
    else:
        data_tasks_groups = None
    calvin_dataset = DiskMultiDataset(
        data_types=data_types,
        datasets_dir=dataset_path,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=window_size,  # 注意这里，假如这里改的时候，后边_build_file_indices_lang也需要改 +1是为了生成下一张图片
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        colour_aug=args.colour_aug,
        data_path_list = args.data_path_list,
        state_matrixs_path = args.state_matrixs_path,
        data_tasks_groups = data_tasks_groups,
        env_resample = args.env_resample,
        only_single_task = args.only_single_task,
        transforms=transforms,
        nav_history_len=args.nav_history_len,
        use_static=use_static,
        SS=SS,
    )

    round_fn = math.floor if floor else math.ceil

    num_samples = len(calvin_dataset) # 总样本数
    global_batch_size = batch_size * args.world_size # 总batch size
    num_batches = round_fn(num_samples / global_batch_size)  # 总iter数
    num_workers = max(1, workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    # sampler = DistributedSampler(
    sampler = DistributedSamplerWithSkip(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    # the batch_size and num_workers are per-GPU !
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collater,
        drop_last=True
    )
    # dataloader = DataLoader(calvin_dataset, batch_size=args.batch_size_calvin)
    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches # 总iter数
    dataloader.num_samples = num_samples # 总样本数
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)



def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    return get_multi_dataset(args, image_processor=image_processor, dataset_type=dataset_type, epoch=epoch, tokenizer=tokenizer)

def load_partial_traj_data():
    with open('partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data

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

if __name__ == "__main__":
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
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    import torchvision.transforms as transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.26862654, 0.26130258, 0.2757771])
    ])

    from transformers import AutoTokenizer

    tokenizer_path = '~/yanfeng/project/robotic/RoboFlamingo/checkpoints/mpt-1b-redpajama-200b-dolly'
    use_local_files = False
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=use_local_files)
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<action>", "<static0>", "<static1>","<static2>","<static3>","<static4>","<static5>","<static6>","<static7>","<static_s>","<static_e>"]}
    )
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<gripper0>", "<gripper1>","<gripper2>","<gripper3>","<gripper4>","<gripper5>","<gripper6>","<gripper7>","<gripper_s>","<gripper_e>"]}
    )
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<obs0>", "<obs1>","<obs2>","<obs3>","<obs4>","<obs5>","<obs6>","<obs7>","<obs_s>","<obs_e>"]}
    )
    text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    # calvin_dataset = get_data(args, transform, text_tokenizer, ["spocvqa"]) # image_processor, tokenizer尝试在此处初始化
    calvin_dataset = get_data(args, transform, text_tokenizer, "calvin")
    # calvin_dataset = get_data(args, transform, text_tokenizer, ["roomvqa"])
    # calvin_dataset.sampler.set_num_skip_iters(100)

    def test():
        for batch_data in tqdm(calvin_dataset.dataloader, desc="Processing batches"):
            pass

    from tqdm import tqdm

    while True:
        try:
            test()
            
        except Exception as e:
            print(e)
            continue