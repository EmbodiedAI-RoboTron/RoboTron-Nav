# these are ordered dicts where the key : value
# is env_name : env_constructor
import argparse
import os
import numpy as np
import pickle
import shutil
from pathlib import Path
from zipfile import ZipFile
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '~/yanfeng/project/robotic/Metaworld/mujoco210'
from PIL import Image
# sys.path.append('~/yanfeng/project/robotic/RoboUniview/third_party/robocasa')
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
from metaworld.policies import *
from typing import List
import open3d as o3d
import math

root_save = '~/yanfeng/data/robotics/metaworld_random3'

all_v2_pol_instance = {'assembly-v2-goal-observable': SawyerAssemblyV2Policy(),
                       'basketball-v2-goal-observable': SawyerBasketballV2Policy(),
                       'bin-picking-v2-goal-observable': SawyerBinPickingV2Policy(),
                       'box-close-v2-goal-observable': SawyerBoxCloseV2Policy(), # 3 1
                       'button-press-topdown-v2-goal-observable': SawyerButtonPressTopdownV2Policy(),
                       'button-press-topdown-wall-v2-goal-observable': SawyerButtonPressTopdownWallV2Policy(),
                       'button-press-v2-goal-observable': SawyerButtonPressV2Policy(),
                       'button-press-wall-v2-goal-observable': SawyerButtonPressWallV2Policy(), # 7 1
                       'coffee-button-v2-goal-observable': SawyerCoffeeButtonV2Policy(),
                       'coffee-pull-v2-goal-observable': SawyerCoffeePullV2Policy(),
                       'coffee-push-v2-goal-observable': SawyerCoffeePushV2Policy(),
                       'dial-turn-v2-goal-observable': SawyerDialTurnV2Policy(),
                       'disassemble-v2-goal-observable': SawyerDisassembleV2Policy(), #12  1
                       'door-close-v2-goal-observable': SawyerDoorCloseV2Policy(),
                       'door-lock-v2-goal-observable': SawyerDoorLockV2Policy(),
                       'door-open-v2-goal-observable': SawyerDoorOpenV2Policy(), # 15 1
                       'door-unlock-v2-goal-observable': SawyerDoorUnlockV2Policy(),
                       'hand-insert-v2-goal-observable': SawyerHandInsertV2Policy(),
                       'drawer-close-v2-goal-observable': SawyerDrawerCloseV2Policy(),
                       'drawer-open-v2-goal-observable': SawyerDrawerOpenV2Policy(),
                       'faucet-open-v2-goal-observable': SawyerFaucetOpenV2Policy(),
                       'faucet-close-v2-goal-observable': SawyerFaucetCloseV2Policy(),
                       'hammer-v2-goal-observable': SawyerHammerV2Policy(), # 22 2
                       'handle-press-side-v2-goal-observable': SawyerHandlePressSideV2Policy(),
                       'handle-press-v2-goal-observable': SawyerHandlePressV2Policy(),
                       'handle-pull-side-v2-goal-observable': SawyerHandlePullSideV2Policy(), # 25 1
                       'handle-pull-v2-goal-observable': SawyerHandlePullV2Policy(),
                       'lever-pull-v2-goal-observable': SawyerLeverPullV2Policy(),
                       'peg-insert-side-v2-goal-observable': SawyerPegInsertionSideV2Policy(), # 28 1
                       'pick-place-wall-v2-goal-observable': SawyerPickPlaceWallV2Policy(), # 29 1
                       'pick-out-of-hole-v2-goal-observable': SawyerPickOutOfHoleV2Policy(),
                       'reach-v2-goal-observable': SawyerReachV2Policy(),
                       'push-back-v2-goal-observable': SawyerPushBackV2Policy(), # 32 1
                       'push-v2-goal-observable': SawyerPushV2Policy(), # 33 1
                       'pick-place-v2-goal-observable': SawyerPickPlaceV2Policy(),
                       'plate-slide-v2-goal-observable': SawyerPlateSlideV2Policy(),
                       'plate-slide-side-v2-goal-observable': SawyerPlateSlideSideV2Policy(), 
                       'plate-slide-back-v2-goal-observable': SawyerPlateSlideBackV2Policy(),
                       'plate-slide-back-side-v2-goal-observable': SawyerPlateSlideBackSideV2Policy(),
                       'peg-unplug-side-v2-goal-observable': SawyerPegUnplugSideV2Policy(), # 39 2
                       'soccer-v2-goal-observable': SawyerSoccerV2Policy(), # 40 1
                       'stick-push-v2-goal-observable': SawyerStickPushV2Policy(),
                       'stick-pull-v2-goal-observable': SawyerStickPullV2Policy(), # 42 1
                       'push-wall-v2-goal-observable': SawyerPushWallV2Policy(),
                       'reach-wall-v2-goal-observable': SawyerReachWallV2Policy(),
                       'shelf-place-v2-goal-observable': SawyerShelfPlaceV2Policy(), # 45 2
                       'sweep-into-v2-goal-observable': SawyerSweepIntoV2Policy(),
                       'sweep-v2-goal-observable': SawyerSweepV2Policy(),
                       'window-open-v2-goal-observable': SawyerWindowOpenV2Policy(),
                       'window-close-v2-goal-observable': SawyerWindowCloseV2Policy()}

# all_view = ('corner', 'corner2', 'corner3', 'topview')
all_view = ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')

def zip_compression(source_dir, target_file):
    with ZipFile(target_file, mode='w') as zf:
        for path, dir_names, filenames in os.walk(source_dir):
            path = Path(path)
            arc_dir = path.relative_to(source_dir)
            for filename in filenames:
                zf.write(path.joinpath(filename), arc_dir.joinpath(filename))


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 
# and combines them into point clouds
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
    def __init__(self, sim, cam_names:List, img_size=84):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_names = cam_names
        
        # List of camera intrinsic matrices
        self.cam_mats = []
        
        for idx in range(len(self.cam_names)):
            # get camera id
            cam_id = self.sim.model.camera_name2id(self.cam_names[idx])
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self, save_img_dir=None, device_id=0):
        o3d_clouds, color_imgs = [], []
        cam_poses = []
        depths = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            color_img, depth = self.captureImage(self.cam_names[cam_i], capture_depth=True, device_id=device_id)
            depths.append(depth)
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
            cam_pos = self.sim.model.body_pos[cam_body_id] + self.sim.model.cam_pos[cam_idx_for_sim]
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_idx_for_sim])
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

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        # get numpy array of point cloud, (position, color)
        combined_cloud_points = np.asarray(combined_cloud.points)
        # color is automatically normalized to [0,1] by open3d
        

        # combined_cloud_colors = np.asarray(combined_cloud.colors)  # Get the colors, ranging [0,1].
        combined_cloud_colors = np.stack(color_imgs).reshape(-1, 3) # range [0, 255]
        combined_cloud = np.concatenate((combined_cloud_points, combined_cloud_colors), axis=1)
        depths = np.array(depths).squeeze()
        return combined_cloud, depths


     
    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
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
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, depth=capture_depth, device_id=device_id)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)

            depth_convert = self.depthimg2Meters(depth)
            img = self.verticalFlip(img)
            return img, depth_convert
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")


def get_point_cloud(pc_generator, use_rgb=True):
    point_cloud, depth = pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
    
    if not use_rgb:
        point_cloud = point_cloud[..., :3]
    
    
    if self.pc_transform is not None:
        point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
    if self.pc_scale is not None:
        point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
    
    if self.pc_offset is not None:    
        point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
    
    if self.use_point_crop:
        if self.min_bound is not None:
            mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
            point_cloud = point_cloud[mask]
        if self.max_bound is not None:
            mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
            point_cloud = point_cloud[mask]

    point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
    
    depth = depth[::-1]
    
    return point_cloud, depth
        
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
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
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
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
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


def deproject(env, cam, rgb_img, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a pixel point to 3D coordinates
    Args
        point: tuple (u, v); pixel coordinates of point to deproject
        depth_img: np.array; depth image used as reference to generate 3D coordinates
        homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                     else returns the world coordinates (x, y, z) position
    Output
        (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
    """
    rgb_img = rgb_img.copy()
    depth_img = depth_img.copy()
    rgb_img = np.flip(rgb_img, axis=0)
    depth_img = np.flip(depth_img, axis=0)
    def depthimg2Meters(depth):
        extent = env.sim.model.stat.extent
        near = env.sim.model.vis.map.znear * extent
        far = env.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image
    depth_img = depthimg2Meters(depth_img)

    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.array(cam['extrinsic_matrix']).reshape((4, 4)) # 此处去掉了求逆
    z = depth_img[v, u]
    foc = cam['cam_config']['height'] / (2 * np.tan(np.deg2rad(cam['cam_config']['fov']) / 2))
    x = (u - cam['cam_config']['width'] // 2) * z / foc
    y = (v - cam['cam_config']['height'] // 2) * z / foc # 注意这里负号删掉了
    z = z  # 注意这里负号删掉了
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.
    if sanity_check:
        sample_inds = np.random.permutation(u.shape[0])[:2000]
        for ind in sample_inds:
            cam_world_pos = cam.deproject((u[ind], v[ind]), depth_img, homogeneous=True)
            assert np.abs(cam_world_pos-world_pos[:, ind]).max() <= 1e-3

    if not homogeneous:
        world_pos = world_pos[:3]

    rgb_img = rgb_img.reshape(-1, 3)/255.0
    world_pos = world_pos.transpose(1,0)

    return world_pos, rgb_img

def main(index):
    image_size=(256, 256) 
    try:
        env_keys = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
        print(env_keys)
        pol_and_env_num = len(all_v2_pol_instance)
        sample_num = 500
        num_eposides = 200 

        env_key = env_keys[index]

        policy = all_v2_pol_instance[env_key]

        for k in range(num_eposides):
            is_succeed=False
            for _ in range(5):
                if is_succeed: break
                env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_key]()
                env._freeze_rand_vec = False
                env.sim.model.vis.map.znear = 0.01
                env.sim.model.vis.map.zfar = 1.5
                env.sim.model.cam_pos[2] = [0.75, 0.075, 0.7]
                obs = env.reset_model()
                obs = env.reset()

                # pc_generator = PointCloudGenerator(sim=env.sim, cam_names=all_view, img_size=image_size[0])
                # point_cloud, depth = pc_generator.generateCroppedPointCloud(device_id=0) # raw point cloud, Nx3

                # print()
                param_list = []
                save_path = f"{root_save}/{env_key}/{env_key}_{k}"
                if os.path.exists(save_path + '.zip'): continue
                mkdir(save_path)
                param_save_path = os.path.join(save_path, 'param')
                mkdir(param_save_path)
                rgb_save_path = os.path.join(save_path, 'rgb')
                mkdir(rgb_save_path)
                print(rgb_save_path)
               
                for j in range(sample_num):
                    
                    a = policy.get_action(obs)
                    a[3] = 0.5 if a[3] > 0 else -0.5
                    a[0:3] = np.clip(a[0:3], -1, 1)

                    obs, reward, done, info = env.step(a)
                    # other param
                    param_dict = {
                        'action': a,
                        'reward': reward,
                        'obs': obs,
                        'is_done': done,
                        'info': info,
                    }
                    param_list.append(param_dict)

                    # rgb
                    rgbs = []
                    clouds = []
                    for view in all_view:
                        rgb_view_save_path = os.path.join(rgb_save_path, view)
                        mkdir(rgb_view_save_path)
                        # rgb = env.render('rgb_array', camera_name=view, resolution=image_size)
                        rgb, depth = env.sim.render(*image_size, camera_name=view, depth=True)

                        p = Image.fromarray(rgb)
                        img_save_path = os.path.join(rgb_view_save_path, f"image_save_{j}.jpg")
                        p.save(img_save_path)
                        depth_save_path = os.path.join(rgb_view_save_path, f"depth_save_{j}.npy")
                        np.save(depth_save_path, depth)
                        if 1: # 得到相机参数
                            # Compute world to camera transformation matrix
                            cam_id = env.sim.model.camera_name2id(view)
                            fov = env.sim.model.cam_fovy[cam_id]
                            # 使用动态外参
                            cam_pos = env.sim.data.cam_xpos[cam_id]
                            c2b_r = rotMatList2NPRotMat(env.sim.data.cam_xmat[cam_id])
                            b2w_r = quat2Mat([0, 1, 0, 0])
                            c2w_r = np.matmul(c2b_r, b2w_r)
                            static_extrinsic = posRotMat2Mat(cam_pos, c2w_r)
                            static_foc = image_size[0] / (2 * np.tan(np.deg2rad(fov) / 2))
                            static_intrinsic = np.array([[static_foc , 0.0, image_size[1]/2], [0.0, static_foc , image_size[0]/2], [0.0, 0.0, 1.0]])
                            cam_config = {'_target_': 'calvin_env.camera.static_camera.StaticCamera', 'name': 'static', 'fov': fov, 'aspect': 1, 
                                'nearval': env.sim.model.vis.map.znear, 'farval': env.sim.model.vis.map.zfar, 'extent': env.sim.model.stat.extent, 'width': image_size[1], 'height': image_size[0], 
                            }
                            cam_params = {
                                'extrinsic_matrix': static_extrinsic,
                                'intrinsic_matrix': static_intrinsic,
                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                'cam_config': cam_config,
                            }
                            config_save_path = os.path.join(rgb_view_save_path, f"config_save_{j}.npy")
                            np.save(config_save_path, cam_params)

                        if 0:
                            clouds, rgbs = deproject(env, cam_params, rgb, depth, homogeneous=False, sanity_check=False)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(clouds[:, :3])
                            pcd.colors = o3d.utility.Vector3dVector(rgbs)
                            o3d.io.write_point_cloud(f'tmp/{j}_{view}.xyz', pcd)

                    if info['success'] > 0:
                        pickle_save_path = os.path.join(param_save_path, 'param.pickle')
                        with open(pickle_save_path, 'wb') as f:
                            pickle.dump(param_list, f)
                        zip_compression(save_path, save_path + '.zip')

                        target_save_path = f"{root_save}/{env_key}/{env_key}_{k}.zip"
                        mkdir(f"{root_save}/{env_key}")
                        shutil.move(save_path + '.zip', target_save_path)
                        shutil.rmtree(save_path, ignore_errors=True)
                        is_succeed = True
                        break

                    if j == 499:
                        shutil.rmtree(save_path)

    except (Exception) as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--tasks', type=int, default=50)
    args = parser.parse_args()
    process_list = []
    pol_and_env_num = len(all_v2_pol_instance)
    # for k in range(0, args.tasks):
    #     main(k)
    # main(1)
    for k in [39]: # 39,45
        main(k)
