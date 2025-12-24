import io, os
import numpy as np
from PIL import Image
import open3d as o3d
import zipfile, pickle
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

abs_datasets_dir = '~/yanfeng/data/robotics/robocasa_uniview'
_task_list = os.listdir(abs_datasets_dir)

def load_from_zip(zip_ref, filename):
    with zip_ref.open(filename) as file:
        return io.BytesIO(file.read())

if 1:
    def process_file(file_idx, zip_file, frames, base_extrinsic):
        
        action = action = frames[file_idx]['action'][:7]

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                with zip_ref.open(os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy")) as file:
                    with io.BytesIO(file.read()) as file_like_object:
                        calib2 = np.load(file_like_object, allow_pickle=True).item()
            ep = {"calib_gripper": calib2}
            gripper_extrinsic_matrix = np.linalg.inv(ep['calib_gripper']['extrinsic_matrix']) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
            gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)
            loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix), np.array([0,0,0,1]))
        except:
            print(file_idx,zip_file)
        if 1:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                rgb = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'image_save_' + str(file_idx) + ".jpg")))
                rgb2 = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg")))
                rgb_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'depth_save_' + str(file_idx) + ".npy")))
                rgb2_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy")))
                calib = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
                calib2 = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
            ep={
                "rgb_static": np.array(rgb),
                "rgb_gripper": np.array(rgb2),
                # "rel_actions": action,
                "robot_obs": frames[file_idx]['states'],
                "scene_obs": frames[file_idx]['states'],
                "depth_static": rgb_depth,
                "depth_gripper": rgb2_depth,
                "calib_static": calib,
                "calib_gripper": calib2,
                }

            rgb, calib = {}, {}
            depth_static = ep['depth_static']
            depth_gripper = ep['depth_gripper']
            static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
            gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
            # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
            static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_extrinsic)
            gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)
    
            static_cam = cam(static_extrinsic_matrix, ep['calib_static']['cam_config']['height'], ep['calib_static']['cam_config']['width'], ep['calib_static']['cam_config']['fov'])
            gripper_cam = cam(gripper_extrinsic_matrix,  ep['calib_gripper']['cam_config']['height'],  ep['calib_gripper']['cam_config']['width'],  ep['calib_gripper']['cam_config']['fov'])
            def depthimg2Meters(depth, cam):
                extent =cam['cam_config']['extent']
                near = cam['cam_config']['nearval'] * extent
                far = cam['cam_config']['farval'] * extent
                image = near / (1 - depth * (1 - near / far))
                return image
            depth_static = np.flip(depth_static, axis=0)
            depth_static = depthimg2Meters(depth_static, ep['calib_static'])
            depth_gripper = np.flip(depth_gripper, axis=0)
            depth_gripper = depthimg2Meters(depth_gripper, ep['calib_gripper'])
            
            static_pcd = deproject(
                static_cam, depth_static,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            gripper_pcd = deproject(
                gripper_cam, depth_gripper,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)

            rgb['rgb_static'] = Image.fromarray(np.flip(ep['rgb_static'], axis=0)) # 注意此处需要添加flip
            rgb['rgb_gripper'] =  Image.fromarray(np.flip(ep['rgb_gripper'], axis=0)) # 注意此处需要添加flip
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
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cloud)
                pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
                o3d.io.write_point_cloud(f"tmp/{file_idx}.pcd", pcd)

        if loc[1] < -0.1 or loc[1] > 0.1:
            print(zip_file)

        return loc, action

    def process_task(task, abs_datasets_dir):
        data_dir = os.path.join(abs_datasets_dir, task)
        if 'navigat' in task: return 
        if not os.path.isdir(data_dir): return 

        for demo_name in [tmp for tmp in os.listdir(data_dir) if '.zip' in tmp]:
            zip_file = os.path.join(data_dir, demo_name)

            zip_file = "~/yanfeng/data/robotics/robocasa_uniview/multi_stage_brewing_PrepareCoffee_2024-05-07/demo_14.zip"

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                with zip_ref.open('param/env.npy') as file:
                    with io.BytesIO(file.read()) as file_like_object:
                        env_param = np.load(file_like_object, allow_pickle=True).item()
                with zip_ref.open('param/param.pickle') as file:
                    with io.BytesIO(file.read()) as file_like_object:
                        frames = pickle.load(file_like_object)
            base_p, base_r = env_param['body_xpos'][1], env_param['body_xmat'][1]
            def rotMatList2NPRotMat(rot_mat_arr):
                np_rot_arr = np.array(rot_mat_arr)
                np_rot_mat = np_rot_arr.reshape((3, 3))
                return np_rot_mat
            def posRotMat2Mat(pos, rot_mat):
                t_mat = np.eye(4)
                t_mat[:3, :3] = rot_mat
                t_mat[:3, 3] = np.array(pos)
                return t_mat

            base_r = rotMatList2NPRotMat(base_r)
            base_extrinsic = posRotMat2Mat(base_p, base_r)

            files = list(range(len(frames)))
            results=[process_file(i, zip_file, frames, base_extrinsic) for i in files]
            for gripper_loc, action in results:
                actions.append(action)
                gripper_locs.append(gripper_loc)
            return 
            # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            #     results = [executor.submit(process_file, i, zip_file, frames, base_extrinsic) for i in files]

            # for future in concurrent.futures.as_completed(results):
            #     gripper_loc, action = future.result()
            #     actions.append(action)
            #     gripper_locs.append(gripper_loc)


    gripper_locs = []
    actions = []
    for task in _task_list:
        process_task(task, abs_datasets_dir)


    gripper_locs = np.stack(gripper_locs)
    actions=np.stack(actions)
    print(gripper_locs.min(0), gripper_locs.max(0))
    print(actions.min(0), actions.max(0))
    np.save('data_folder/robocasa_gripper_limit.npy', gripper_locs)
    np.save('data_folder/robocasa_actions.npy', actions)
    for i in range(3):
        print(plt.hist(gripper_locs[:, i], bins=20, color='blue', edgecolor='black'))
    for i in range(12):
        print(plt.hist(actions[:, i], bins=20, color='blue', edgecolor='black'))


if 0:
    gripper_locs = []
    for task in _task_list:
        if 'navigat' in task: continue
        # if 'kitchen_doors_' not in task: continue
        data_dir = os.path.join(abs_datasets_dir, task)
        if not os.path.isdir(data_dir): continue # 样本均在文件夹内，假如非文件夹则无效
        for demo_name in [tmp for tmp in os.listdir(data_dir) if '.zip' in tmp]:
            zip_file = os.path.join(data_dir, demo_name)
            # zip_file = "~/yanfeng/data/robotics/robocasa_uniview/single_stage_kitchen_drawer_CloseDrawer_2024-04-30/demo_1.zip"
            # print(zip_file)
            
            env_param=np.load(io.BytesIO(ZipReader.read(zip_file, 'param/env.npy')), allow_pickle=True).item()
            base_p, base_r = env_param['body_xpos'][1], env_param['body_xmat'][1]
            def rotMatList2NPRotMat(rot_mat_arr):
                np_rot_arr = np.array(rot_mat_arr)
                np_rot_mat = np_rot_arr.reshape((3, 3))
                return np_rot_mat
            def posRotMat2Mat(pos, rot_mat):
                t_mat = np.eye(4)
                t_mat[:3, :3] = rot_mat
                t_mat[:3, 3] = np.array(pos)
                return t_mat
            base_r = rotMatList2NPRotMat(base_r)
            base_extrinsic = posRotMat2Mat(base_p, base_r)

            frames = pickle.load(io.BytesIO(ZipReader.read(zip_file, 'param/param.pickle')))
            for file_idx in range(len(frames)):
           
                # calib = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
                calib2 = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
                ep = {
                    # "calib_static": calib,
                    "calib_gripper": calib2,
                }

                # static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
                gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
                # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
                # static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_extrinsic)
                gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)

                # # 平移矩阵 T_translate
                # T_translate = np.array([
                #     [1, 0, 0, 0.45], # [0.1, 0.8] - [-0.5, 0.5]
                #     [0, 1, 0, 0.0], # [-0.8, 0.8]
                #     [0, 0, 1, 0.6], # [0.9. 1.5]
                #     [0, 0, 0, 1]
                # ])
                # static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
                # gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
                
                loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix),np.array([0,0,0,1]))
                gripper_locs.append(loc)
                # if loc[0] < -0.3:
                #     print(loc)

            # break

    gripper_locs=np.stack(gripper_locs)
    np.save('data_dolder/robocasa_gripper_limit.py', gripper_locs)
    print(gripper_locs.min(0), gripper_locs.max(0))
    for i in range(3):
        print(plt.hist(gripper_locs[:,i], bins=20, color='blue', edgecolor='black'))

if 0:
    for task in _task_list:
        if 'navigat' in task: continue
        if 'kitchen_doors_' not in task: continue
        data_dir = os.path.join(abs_datasets_dir, task)
        if not os.path.isdir(data_dir): continue # 样本均在文件夹内，假如非文件夹则无效
        for demo_name in [tmp for tmp in os.listdir(data_dir) if '.zip' in tmp]:
            zip_file = os.path.join(data_dir, demo_name)
            # zip_file = "~/yanfeng/data/robotics/robocasa_uniview/single_stage_kitchen_drawer_CloseDrawer_2024-04-30/demo_1.zip"
            print(zip_file)
            
            env_param=np.load(io.BytesIO(ZipReader.read(zip_file, 'param/env.npy')), allow_pickle=True).item()
            base_p, base_r = env_param['body_xpos'][1], env_param['body_xmat'][1]
            def rotMatList2NPRotMat(rot_mat_arr):
                np_rot_arr = np.array(rot_mat_arr)
                np_rot_mat = np_rot_arr.reshape((3, 3))
                return np_rot_mat
            def posRotMat2Mat(pos, rot_mat):
                t_mat = np.eye(4)
                t_mat[:3, :3] = rot_mat
                t_mat[:3, 3] = np.array(pos)
                return t_mat
            base_r = rotMatList2NPRotMat(base_r)
            base_extrinsic = posRotMat2Mat(base_p, base_r)

            frames = pickle.load(io.BytesIO(ZipReader.read(zip_file, 'param/param.pickle')))
            for file_idx in range(len(frames)):
                
                action = frames[file_idx]['action'][:7] # 注意这里移动12个值，前6个是爪子位姿，第7个是爪子张合，第8-10是移动，11是躯干，12是底盘移动还是夹爪

                # ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand", "robot0_agentview_center", "robot0_frontview"],
                rgb = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'image_save_' + str(file_idx) + ".jpg"))))
                rgb2 = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg"))))
                rgb_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'depth_save_' + str(file_idx) + ".npy"))))
                rgb2_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy"))))
                calib = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
                calib2 = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
                ep = {
                    "rgb_static": np.array(rgb),
                    "rgb_gripper": np.array(rgb2),
                    "rel_actions": action,
                    "robot_obs": frames[file_idx]['states'],
                    "scene_obs": frames[file_idx]['states'],
                    "depth_static": rgb_depth,
                    "depth_gripper": rgb2_depth,
                    "calib_static": calib,
                    "calib_gripper": calib2,
                }

                rgb, calib = {}, {}
                depth_static = ep['depth_static']
                depth_gripper = ep['depth_gripper']
                static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
                gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
                # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
                static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_extrinsic)
                gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)

                static_cam = cam(static_extrinsic_matrix, ep['calib_static']['cam_config']['height'], ep['calib_static']['cam_config']['width'], ep['calib_static']['cam_config']['fov'])
                gripper_cam = cam(gripper_extrinsic_matrix,  ep['calib_gripper']['cam_config']['height'],  ep['calib_gripper']['cam_config']['width'],  ep['calib_gripper']['cam_config']['fov'])
                def depthimg2Meters(depth, cam):
                    extent =cam['cam_config']['extent']
                    near = cam['cam_config']['nearval'] * extent
                    far = cam['cam_config']['farval'] * extent
                    image = near / (1 - depth * (1 - near / far))
                    return image
                depth_static = np.flip(depth_static, axis=0)
                depth_static = depthimg2Meters(depth_static, ep['calib_static'])
                depth_gripper = np.flip(depth_gripper, axis=0)
                depth_gripper = depthimg2Meters(depth_gripper, ep['calib_gripper'])
                
                static_pcd = deproject(
                    static_cam, depth_static,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                gripper_pcd = deproject(
                    gripper_cam, depth_gripper,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)

                rgb['rgb_static'] = Image.fromarray(np.flip(ep['rgb_static'], axis=0)) # 注意此处需要添加flip
                rgb['rgb_gripper'] =  Image.fromarray(np.flip(ep['rgb_gripper'], axis=0)) # 注意此处需要添加flip        rgb['rgb_static'] = np.array(rgb['rgb_static'])
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
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
                    o3d.io.write_point_cloud(f"tmp/{task}_{demo_name}_{file_idx}.pcd", pcd)

            # break

    gripper_locs=np.stack(gripper_locs)
    print(gripper_locs.min(0), gripper_locs.max(0))

def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat
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
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat
