import sys

sys.path.append("~/yanfeng/project/robotic/LIBERO/")
import argparse
from collections import Counter, defaultdict, namedtuple
import logging
import os, json, random
import robosuite.utils.transform_utils as T
from pathlib import Path
import sys, gc
import time
import PIL.Image as Image
import copy
from tqdm.auto import tqdm
from collections import deque
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset, get_dataset)
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
import torch
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip
import open3d as o3d
from pathlib import Path

def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat



def generate_data(args):
    # generate the data we need

    # 导入仿真环境，每个task的数据单独生成
    env_num = 1
    eval_loop_num = 20
    img_h, img_w = 256, 256

    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")
    task_order = 0  # can be from {0 .. 21}, default to 0, which is [task 0, 1, 2 ...]

    benchmark_name = "libero_90"  # can be from {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    benchmark = get_benchmark(benchmark_name)(task_order)
    n_tasks = benchmark.n_tasks

    task_list = list(range(n_tasks))

    task_sequences = tqdm(task_list, position=0, leave=True)

    datasets = []
    descriptions = []
    shape_meta = None
    abs_datasets_dir = '~/yanfeng/data/robotics/LIBERO'

    save_dir = args.save_dir
    modality = {"rgb": ["agentview_rgb", "eye_in_hand_rgb"],
                "depth": [],
                "low_dim": ["gripper_states", "joint_states"]}

    os.makedirs(save_dir, exist_ok=True)

    home = Path(save_dir)
    for task_id in task_sequences:
        os.makedirs(home / f'task_{task_id}', exist_ok=True)

        # 取出所有帧
        task_i_dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(abs_datasets_dir, benchmark.get_task_demonstration(task_id)),
            obs_modality=modality,
            initialize_obs_utils=True,
            dataset_keys=['states', 'actions']
        )

        description = benchmark.get_task(task_id).language
        dataset = task_i_dataset

        dataset = SequenceVLDataset(task_i_dataset, description)
        n_demos = dataset.n_demos
        n_sequences = dataset.total_num_sequences
        # 获取demo长度
        demo_length = np.cumsum(list(dataset.sequence_dataset._demo_id_to_demo_length.values()))
        task = benchmark.get_task(task_id)
        demo_length = np.insert(demo_length, 0, 0)

        env_args = {
            "bddl_file_name": os.path.join(
                bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": img_h,
            "camera_widths": img_w,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_depths": True,
        }

        env = OffScreenRenderEnv(
            **env_args
        )
        env.reset()
        for demo in range(len(demo_length) - 1):

            os.makedirs(home / f"task_{task_id}" / f"demo_{demo}", exist_ok=True)
            demo_path = home / f"task_{task_id}" / f"demo_{demo}"
            # frame_list = []

            # 获取数据集中各物体状态
            for step in range(demo_length[demo], demo_length[demo + 1]):
                camera_dict = {}
                states = dataset.__getitem__(step)['states'][0]
                camera_dict['action'] = dataset.__getitem__(step)['actions'][0]
                camera_dict['states'] = states
                env.set_init_state(states)
                env.sim.model.vis.map.znear = 0.01
                env.sim.model.vis.map.zfar = 1.5
                camera_dict['base_p'] = env.sim.data.body_xpos[2],
                camera_dict['base_r'] = env.sim.data.body_xmat[2]
                for view in env_args['camera_names']:
                    camera_dict[view] = get_camera_conf(env, [view], 256)
                np.savez(demo_path / f"{step - demo_length[demo]}.npz", **camera_dict)

            # img_clip = ImageSequenceClip(frame_list, fps=30)
            # img_clip.write_gif('data.gif', fps=30)
            # task = benchmark.get_task(task_id)

            # env.reset()
            # for step in range(len(action_list)):
            #     env.set_init_state(state_list[step])
            #     camera_dict = {}
            #     for view in env_args['camera_names']:
            #         camera_dict[view] = get_camera_conf(env, [view], 128)
            #     camera_dict['action'] = action_list[step]
            #     camera_dict['states'] = state_list[step]
            #     # 获取当前情况

            # np.savez(demo_path / f"{step}.npz", **camera_dict)

            # img_clip = ImageSequenceClip(env_list, fps=30)
            # img_clip.write_gif(f'env_{demo}.gif', fps=30)
            # task = benchmark.get_task(task_id)


def get_camera_conf(env, cam_names=None, img_size=256):
    pc_generator = PointCloudGenerator(sim=env.sim, cam_names=cam_names, img_size=img_size)

    # point_cloud, depth, depth_color, origin_depth = pc_generator.generateCroppedPointCloud()
    #
    # cloud = o3d.geometry.PointCloud()
    #
    # # 将 NumPy 数组转换为 open3d 的 Vector3dVector
    # cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # cloud.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    #
    # # 保存点云数据为 PCD 文件
    # o3d.io.write_point_cloud("example_with_colors.pcd", cloud)

    # point_cloud, depth, depth_color, origin_depth = pc_generator.get_cameras_config()
    cam_dict = pc_generator.get_cameras_config()

    calib = {
        'extrinsic_matrix': pc_generator.cam_extrinsic[0],
        'intrinsic_matrix': pc_generator.cam_intrinsic[0],
        'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
    }

    cam_dict['calib'] = calib
    # return point_cloud, depth, calib, depth_color, origin_depth
    return cam_dict


"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""


class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """

    def __init__(self, sim, cam_names, img_size=256):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_names = cam_names

        self.cam_intrinsic = []
        self.cam_extrinsic = []
        self.fov = []
        # List of camera intrinsic matrices
        self.cam_mats = []

        for idx in range(len(self.cam_names)):
            # get camera id
            cam_id = self.sim.model.camera_name2id(self.cam_names[idx])
            fovy = np.deg2rad(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * np.tan(fovy / 2))
            # fovy = np.deg2rad(math.radians(self.sim.model.cam_fovy[cam_id]))
            # f = self.img_height / (2 * np.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

            self.cam_intrinsic.append(cam_mat)
            self.fov.append(self.sim.model.cam_fovy[cam_id])

    def generateCroppedPointCloud(self, save_img_dir=None, device_id=0):
        o3d_clouds = []
        color_imgs = []
        cam_poses = []

        depths = []
        origin_depths = []

        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            color_img, depth, orgin_depth = self.captureImage(self.cam_names[cam_i], capture_depth=True,
                                                              device_id=device_id)
            depths.append(depth)
            origin_depths.append(orgin_depth)

            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if save_img_dir != None:
                self.saveImg(depth, save_img_dir, "depth_test_" + str(cam_i))
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud

            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)

            # od_color = o3d.geometry.Image(color_img)  # Convert the color image to Open3D format
            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(od_color, od_depth)  # Create an RGBD image
            # o3d_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            #     rgbd_image,
            #     od_cammat)

            # Compute world to camera transformation matrix
            cam_idx_for_sim = self.sim.model.camera_name2id(self.cam_names[cam_i])
            cam_body_id = self.sim.model.cam_bodyid[cam_idx_for_sim]
            cam_pos = self.sim.data.cam_xpos[
                cam_idx_for_sim]  # self.sim.model.body_pos[cam_body_id] + self.sim.model.cam_pos[cam_idx_for_sim]
            c2b_r = rotMatList2NPRotMat(self.sim.data.cam_xmat[cam_idx_for_sim])
            # c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_idx_for_sim])
            # print("c2b_r", c2b_r)
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)
            o3d_clouds.append(transformed_cloud)
            color_imgs.append(color_img)

            self.cam_extrinsic.append(c2w)
        # print("self.cam_extrinsic", self.cam_extrinsic)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        # get numpy array of point cloud, (position, color)
        combined_cloud_points = np.asarray(combined_cloud.points)
        # color is automatically normalized to [0,1] by open3d

        # combined_cloud_colors = np.asarray(combined_cloud.colors)  # Get the colors, ranging [0,1].
        combined_cloud_colors = np.stack(color_imgs).reshape(-1, 3)  # range [0, 255]
        combined_cloud = np.concatenate((combined_cloud_points, combined_cloud_colors), axis=1)

        depths = np.array(depths).squeeze()
        depths_color = np.stack(color_imgs).squeeze()
        origin_depths = np.array(origin_depths).squeeze()

        return combined_cloud, depths, depths_color, origin_depths

    def get_cameras_config(self, save_img_dir=None, device_id=0):
        color_imgs = []
        cam_poses = []

        depths = []
        origin_depths = []

        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            color_img, depth, orgin_depth = self.captureImage(self.cam_names[cam_i], capture_depth=True,
                                                              device_id=device_id)
            depths.append(depth)
            origin_depths.append(orgin_depth)

            # Compute world to camera transformation matrix
            cam_idx_for_sim = self.sim.model.camera_name2id(self.cam_names[cam_i])
            cam_body_id = self.sim.model.cam_bodyid[cam_idx_for_sim]
            cam_pos = self.sim.data.cam_xpos[
                cam_idx_for_sim]  # self.sim.model.body_pos[cam_body_id] + self.sim.model.cam_pos[cam_idx_for_sim]
            c2b_r = rotMatList2NPRotMat(self.sim.data.cam_xmat[cam_idx_for_sim])
            # c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_idx_for_sim])
            # print("c2b_r", c2b_r)
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)

            color_imgs.append(color_img)

            self.cam_extrinsic.append(c2w)

        depths = np.array(depths).squeeze()
        depths_color = np.stack(color_imgs).squeeze()
        origin_depths = np.array(origin_depths).squeeze()

        return {'depth': depths, 'depth_color': depths_color, 'origin_depth': origin_depths, 'image': color_img}

    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, camera_name, capture_depth=True, device_id=0):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, depth=capture_depth,
                                          device_id=device_id)
        if capture_depth:
            img, depth = rendered_images
            orgin_depth = depth
            depth = self.verticalFlip(depth)
            depth_convert = self.depthimg2Meters(depth)
            img = self.verticalFlip(img)

            return img, depth_convert, orgin_depth
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file


# 四元数转旋转矩阵
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
                   2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
                   2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat


"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""


def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
                        default='~/yanfeng/data/robotics/libero_uniview_final')
    args = parser.parse_args()
    generate_data(args)