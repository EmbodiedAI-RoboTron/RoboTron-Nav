import os, sys
import json
import h5py
import argparse
import imageio
import numpy as np
import random
import time
import shutil
import pickle
import re
from pathlib import Path
from PIL import Image
import open3d as o3d
from termcolor import colored
os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '~/yanfeng/project/robotic/Metaworld/mujoco210'
sys.path.append('~/yanfeng/project/robotic/RoboUniview/third_party/robocasa')
sys.path.append('~/yanfeng/project/robotic/robosuite')
import robocasa
import robosuite_casa as robosuite
from zipfile import ZipFile

root_save = '~/yanfeng/data/robotics/robocasa_uniview_tmp'

     
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


def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
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
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix))
    z = depth_img[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos

class cam:
    """cam object class for calvin"""
    def __init__(self,viewMatrix,height,width,fov):
        self.viewMatrix = viewMatrix
        self.height = height
        self.width = width
        self.fov = fov

def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    actions=None, 
    render=False, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
    verbose=False,
    save_path='tmp',
    image_size=[256,256]
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)
    env.sim.model.vis.map.znear = 0.01 # YF add
    env.sim.model.vis.map.zfar = 1.5

    traj_len = states.shape[0]
    action_playback = False
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    param_list = []
    # return
    # save_path = f"{root_save}/{env_key}/{env_key}_{k}"
    shutil.rmtree(save_path, ignore_errors=True)
    if os.path.exists(save_path + '.zip'): return 
    mkdir(save_path)
    param_save_path = os.path.join(save_path, 'param')
    mkdir(param_save_path)
    rgb_save_path = os.path.join(save_path, 'rgb')
    mkdir(rgb_save_path)
    env_param = {"initial_state": initial_state, "body_xmat": env.sim.data.body_xmat, "body_xpos": env.sim.data.body_xpos}

    for i in range(traj_len):
        start = time.time()

        if action_playback:
            env.step(actions[i])
            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = env.get_state()["states"]
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    print("warning: playback diverged by {} at step {}".format(err, i))
        else:
            reset_to(env, {"states" : states[i]})

        # on-screen render
        if render:
            # env.render(mode="human", camera_name=camera_names[0])
            if env.viewer is None:
                env.initialize_renderer()
                
            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        param_dict = {
            'action': actions[i],
            'states': states[i],
        }
        param_list.append(param_dict)

        for cam_name in camera_names:
            rgb_view_save_path = os.path.join(rgb_save_path, cam_name)
            mkdir(rgb_view_save_path)

            rgb, depth = env.sim.render(*image_size, camera_name=cam_name, depth=True)

            p = Image.fromarray(rgb)
            img_save_path = os.path.join(rgb_view_save_path, f"image_save_{i}.jpg")
            p.save(img_save_path)
            depth_save_path = os.path.join(rgb_view_save_path, f"depth_save_{i}.npy")
            np.save(depth_save_path, depth)

            if 1: # 得到相机参数
                # Compute world to camera transformation matrix
                cam_id = env.sim.model.camera_name2id(cam_name)
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
                config_save_path = os.path.join(rgb_view_save_path, f"config_save_{i}.npy")
                np.save(config_save_path, cam_params)

            if 0:

                static_extrinsic_matrix = np.linalg.inv(cam_params['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
                static_cam = cam(static_extrinsic_matrix, cam_params['cam_config']['height'], cam_params['cam_config']['width'], cam_params['cam_config']['fov'])
                def depthimg2Meters(depth, cam):
                    extent =cam['cam_config']['extent']
                    near = cam['cam_config']['nearval'] * extent
                    far = cam['cam_config']['farval'] * extent
                    image = near / (1 - depth * (1 - near / far))
                    return image
                depth_static = np.flip(depth, axis=0)
                depth_static = depthimg2Meters(depth_static, cam_params)
                static_pcd = deproject(
                                        static_cam, depth_static,
                                        homogeneous=False, sanity_check=False
                                    ).transpose(1, 0)
                static_rgb = np.flip(rgb, axis=0)
                static_rgb =  np.reshape(
                    static_rgb, ( static_rgb.shape[0]*static_rgb.shape[1], 3)
                )
                static_rgb = static_rgb/255
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(static_rgb)
                o3d.io.write_point_cloud(f'tmp/{i}_{cam_name}.pcd', pcd)


        # video render
        # if write_video:
        #     if video_count % video_skip == 0:
        #         video_img = []
        #         for cam_name in camera_names:
        #             # video_img.append(env.sim.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
        #             video_img.append(env.sim.render(height=512, width=512, camera_name=cam_name))
        #         video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
        #         video_writer.append_data(video_img)
        #     video_count += 1

        # if first:
        #     break

    pickle_save_path = os.path.join(param_save_path, 'param.pickle')
    with open(pickle_save_path, 'wb') as f:
        pickle.dump(param_list, f)

    # json.loads(initial_state["ep_meta"])
    # lang = ep_meta.get("lang", None)
    np.save(os.path.join(param_save_path, 'env.npy'), env_param)
    zip_compression(save_path, save_path + '.zip')

    # target_save_path = f"{root_save}/{env_key}/{env_key}_{k}.zip"
    # mkdir(f"{root_save}/{env_key}")
    # shutil.move(save_path + '.zip', target_save_path)
    shutil.rmtree(save_path, ignore_errors=True)
    # is_succeed = True

def playback_trajectory_with_obs(
    traj_grp,
    video_writer, 
    video_skip=5, 
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break

def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    elif ds_format == "r2d2":
        env_meta = dict(f.attrs)
    else:
        raise ValueError
    f.close()
    return env_meta

class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """
    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(f"ObservationKeyToModalityDict: {item} not found,"
                  f" adding {item} to mapping with assumed low_dim modality!")
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)
    

def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml
    
    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"): # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"): # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        # 需要替换掉里边的路径
        xml = xml.replace('/Users/lancezhang04/robocasa/robosuite-dev/robosuite', '~/yanfeng/project/robotic/robosuite/robosuite_casa')
        xml = xml.replace('/Users/aaronlo/Desktop/rpl3/robosuite-dev/robosuite', '~/yanfeng/project/robotic/robosuite/robosuite_casa')
        xml = xml.replace('/Users/abhishek/Documents/research/robocasa/robosuite-dev/robosuite', '~/yanfeng/project/robotic/robosuite/robosuite_casa')
        xml = re.sub(r'/Users/.*?/robosuite-dev/robosuite', r'~/yanfeng/project/robotic/robosuite/robosuite_casa', xml)
        xml = re.sub(r'/home/.*?/robosuite-dev/robosuite', r'~/yanfeng/project/robotic/robosuite/robosuite_casa', xml)

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if hasattr(env, "unset_ep_meta"): # unset the ep meta after reset complete
            env.unset_ep_meta()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def playback_dataset(args, env_key):
    # some arg checking
    write_video = True #(args.video_path is not None)
    # write_video = False
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".mp4")[0] + "_use_actions.mp4"
    assert not (args.render and write_video) # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = "robot0_agentview_center"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    env = None
   
    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        # initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        # env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False # absolute action space

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["has_renderer"] = True
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True #w rite_video
        env_kwargs["use_camera_obs"] = False

        if args.verbose:
            print(colored("Initializing environment for {}...".format(env_kwargs["env_name"]), "yellow"))
        
        env = robosuite.make(**env_kwargs)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    elif "data" in f.keys():
        demos = list(f["data"].keys())
    else:
        demos = None

    if demos is not None:
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
    else:
        """rendering for r2d2"""
        assert args.use_obs
        video_writer = None
        if write_video:
            video_writer = imageio.get_writer(args.video_path, fps=20)
        playback_trajectory_with_obs(
            traj_grp=f, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            image_names=args.render_image_names,
            first=args.first,
        )
        f.close()
        if write_video:
            video_writer.close()
        return
    
    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        # if not args.dont_shuffle_demos:
        #     random.shuffle(demos)
        random.shuffle(demos)
        demos = demos[:args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        # ind = len(demos)-1-ind
        ep = demos[ind]
        print(colored("Playing back episode: {}".format(ep), "yellow"))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        # if args.extend_states:
        #     states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        if True: # args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
            # actions = f["data/{}/actions_abs".format(ep)][()] # absolute actions
        save_path = f"{root_save}/{env_key}/{ep}"
        print(save_path)
        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            verbose=args.verbose,
            save_path=save_path,
        )

        # break

    f.close()
    if write_video:
        video_writer.close()

    if env is not None:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='~/yanfeng/project/robotic/RoboUniview/third_party/robocasa/../../../../../data/robotics/robocasa/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo.hdf5',
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default="tmp.gif", #只支持gif
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand", "robot0_agentview_center", "robot0_frontview"],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--verbose",
        action='store_true',
        default=True,
        help="log additional information",
    )

    args = parser.parse_args()

    root = '~/yanfeng/data/robotics/robocasa/v0.1'
    for stage in ['single_stage', 'multi_stage']:
        stage_path = f"{root}/{stage}"
        for scence in os.listdir(stage_path):
            # if scence != "kitchen_coffee": continue
            # if scence != "kitchen_doors": continue
            # if scence != "kitchen_drawer": continue
            # if scence != "kitchen_microwave": continue
            # if scence != "kitchen_navigate": continue
            # if scence != "kitchen_pnp": continue
            # if scence != "kitchen_sink": continue
            # if scence != "kitchen_stove": continue
            # if scence != "brewing": continue
            # if scence != "chopping_food": continue
            # if scence != "defrosting_food": continue
            # if scence != "restocking_supplies": continue
            # if scence != "washing_dishes": continue

            scence_path = f"{stage_path}/{scence}"
            for task in os.listdir(scence_path):
                task_path = f"{scence_path}/{task}"
                if task != "PnPStoveToCounter": continue
                # if task != "OpenDoubleDoor": continue
                for demo in os.listdir(task_path):
                    demo_path = f"{task_path}/{demo}/demo.hdf5"
                    args.dataset = demo_path
                    
                    # for path in 
                    playback_dataset(args, f"{stage}_{scence}_{task}_{demo}")
