import json
import io, os
import re,h5py
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
import zipfile, pickle
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

abs_datasets_dir = Path('~/yanfeng/data/robotics/task_D_D/training')
data_path = '~/liufanfan-mv-from/data/new_calvin_D_1/training'
lang_data = np.load(abs_datasets_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True,).item()
if 1:
     
    def angle_between_angles(a, b):
        diff = b - a
        return  (diff + np.pi) % (2 * np.pi) - np.pi #  (diff + np.pi) % (2 * np.pi) - np.pi
    def process_file(file_idx, data_path):
        filename = f"{data_path}/episode_{file_idx:0{7}d}.npz"
        try:
            ep = np.load(filename, allow_pickle=True)
            action = np.hstack([angle_between_angles(ep['robot_obs'][:3], ep['actions'][:3]), angle_between_angles(ep['robot_obs'][3:6],ep['actions'][3:6]), ep['actions'][-1:]])  # =ep['rel_actions']

            if 1:
                rgb={}
                calib = ep['calib'].item()
                cam_config = ep['cam_config'].item()
                depth_static = ep['depth_static']
                depth_gripper = ep['depth_gripper']
                static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
                gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']
                # 平移矩阵 T_translate # 原始是0.3-0.8因此无需添加平移矩阵
                static_cam = cam(static_extrinsic_matrix,cam_config['static']['height'],cam_config['static']['width'],cam_config['static']['fov'])
                gripper_cam = cam(gripper_extrinsic_matrix,cam_config['gripper']['height'],cam_config['gripper']['width'],cam_config['gripper']['fov'])
                static_pcd = deproject(
                    static_cam, depth_static,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)

                gripper_pcd = deproject(
                    gripper_cam, depth_gripper,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                cloud = np.concatenate([static_pcd,gripper_pcd],axis=0)
                rgb['rgb_static'] = Image.fromarray(ep['rgb_static'])
                rgb['rgb_gripper'] =  Image.fromarray(ep['rgb_gripper'])
                rgb['rgb_static'] = np.array(rgb['rgb_static'])
                rgb['rgb_gripper'] = np.array(rgb['rgb_gripper'])
                static_rgb =  np.reshape(
                    rgb['rgb_static'], ( rgb['rgb_static'].shape[0]*rgb['rgb_static'].shape[1], 3)
                )
                gripper_rgb =  np.reshape(
                    rgb['rgb_gripper'], (rgb['rgb_gripper'].shape[0]*rgb['rgb_gripper'].shape[1], 3)
                )
                pcd_rgb = np.concatenate([static_rgb,gripper_rgb],axis=0)
                pcd_rgb = pcd_rgb/255
                if 1:
                    _pcd = o3d.geometry.PointCloud()
                    _pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    _pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
                    o3d.io.write_point_cloud("tmp.pcd", _pcd)
            extrinsic_matrix = ep['calib'].item()['rgb_gripper']['extrinsic_matrix']
            ep = {
                "rgb_gripper": {"extrinsic_matrix": extrinsic_matrix},
            }
            gripper_extrinsic_matrix = ep['rgb_gripper']['extrinsic_matrix']
            # gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)
            gripper_loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix), np.array([0,0,0,1]))
            return action, gripper_loc
        except:
            return None, None

    gripper_locs = []
    actions = []
    ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64, me
    for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
        print(i)
        files = list(range(start_idx, end_idx + 1))
        results = [process_file(i, data_path) for i in files]
        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #     results = [executor.submit(process_file, i, data_path) for i in files]

        # for future in concurrent.futures.as_completed(results):
        #     action, gripper_loc = future.result()
        for action, gripper_loc in results:
            if action is None: continue
            actions.append(action)
            gripper_locs.append(gripper_loc)

    gripper_locs=np.stack(gripper_locs)
    actions=np.stack(actions)
    print(gripper_locs.min(0), gripper_locs.max(0))
    print(actions.min(0), actions.max(0))
    np.save('data_folder/calvin_gripper_limit.py', gripper_locs)
    np.save('data_folder/calvin_actions.py', actions)
    for i in range(3):
        print(plt.hist(gripper_locs[:,i], bins=20, color='blue', edgecolor='black'))
    for i in range(7):
        print(plt.hist(actions[:,i], bins=20, color='blue', edgecolor='black'))


def show(env, filename):

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
    if 1:
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
        o3d.io.write_point_cloud(f"tmp/{filename}0.pcd", pcd)
        pcd.points = o3d.utility.Vector3dVector(gripper_pcd[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(rgb_gripper)
        o3d.io.write_point_cloud(f"tmp/{filename}1.pcd", pcd)
