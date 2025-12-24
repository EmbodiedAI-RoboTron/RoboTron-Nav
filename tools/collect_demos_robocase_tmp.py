"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
import datetime
import imageio
import json
import os, sys
import time
from glob import glob
from termcolor import colored
os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '~/yanfeng/project/robotic/Metaworld/mujoco210'
import h5py
import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

sys.path.append('~/yanfeng/project/robotic/RoboUniview/third_party/robocasa')
import robocasa
import robocasa.macros as macros
from robocasa.models.objects.fixtures import FixtureType

import mujoco
assert mujoco.__version__ == "3.1.1", "MuJoCo version must be 3.1.1. Please run pip install mujoco==3.1.1"

import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

import init_path
import libero.libero.utils.utils as libero_utils
import cv2
from PIL import Image
from robosuite.utils import camera_utils

from libero.libero.envs import *
from libero.libero import get_libero_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", default="demo.hdf5")

    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument("--use-camera-obs", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="training_set",
    )

    parser.add_argument("--no-proprio", action="store_true")

    parser.add_argument(
        "--use-depth",
        action="store_true",
    )

    args = parser.parse_args()

    hdf5_path = "~/yanfeng/data/robotics/robocasa/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams.hdf5"
    f = h5py.File(hdf5_path, "r")
    hdf5_path = "~/yanfeng/data/robotics/LIBERO/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5"
    f_libero = h5py.File(hdf5_path, "r")

    env_name = f["data"].attrs["env"]

    env_args = f["data"].attrs["env_info"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])

    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_info["domain_name"]
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info["language_instruction"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    bddl_file_name = f["data"].attrs["bddl_file_name"]

    bddl_file_dir = os.path.dirname(bddl_file_name)
    replace_bddl_prefix = "/".join(bddl_file_dir.split("bddl_files/")[:-1] + "bddl_files")

    hdf5_path = os.path.join(get_libero_path("datasets"), bddl_file_dir.split("bddl_files/")[-1].replace(".bddl", "_demo.hdf5"))

    output_parent_dir = Path(hdf5_path).parent
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    h5py_f = h5py.File(hdf5_path, "w")

    grp = h5py_f.create_group("data")

    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
        camera_names=[
            "robot0_eye_in_hand",
            "agentview",
        ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )

    grp.attrs["bddl_file_name"] = bddl_file_name
    grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
    print(grp.attrs["bddl_file_content"])

    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )

    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": f["data"].attrs["bddl_file_name"],
        "env_kwargs": env_kwargs,
    }

    grp.attrs["env_args"] = json.dumps(env_args)
    print(grp.attrs["env_args"])
    total_len = 0
    demos = demos

    cap_index = 5

    for (i, ep) in enumerate(demos):
        print("Playing back random episode... (press ESC to quit)")

        # # select an episode randomly
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        reset_success = False
        while not reset_success:
            try:
                env.reset()
                reset_success = True
            except:
                continue

        model_xml = libero_utils.postprocess_model_xml(model_xml, {})

        if not args.use_camera_obs:
            env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        actions = np.array(f["data/{}/actions".format(ep)][()])

        num_actions = actions.shape[0]

        init_idx = 0
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states[init_idx])
        env.sim.forward()
        model_xml = env.sim.model.get_xml()

        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []

        agentview_images = []
        eye_in_hand_images = []

        agentview_depths = []
        eye_in_hand_depths = []

        agentview_seg = {0: [], 1: [], 2: [], 3: [], 4: []}

        rewards = []
        dones = []

        valid_index = []

        for j, action in enumerate(actions):

            obs, reward, done, info = env.step(action)

            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                # assert(np.all(np.equal(states[j + 1], state_playback)))
                err = np.linalg.norm(states[j + 1] - state_playback)

                if err > 0.01:
                    print(
                        f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}"
                    )

            # Skip recording because the force sensor is not stable in
            # the beginning
            if j < cap_index:
                continue

            valid_index.append(j)

            if not args.no_proprio:
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])

                joint_states.append(obs["robot0_joint_pos"])

                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )

            robot_states.append(env.get_robot_state_vector(obs))

            if args.use_camera_obs:

                if args.use_depth:
                    agentview_depths.append(obs["agentview_depth"])
                    eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
            else:
                env.render()

        # end of one trajectory
        states = states[valid_index]
        actions = actions[valid_index]
        dones = np.zeros(len(actions)).astype(np.uint8)
        dones[-1] = 1
        rewards = np.zeros(len(actions)).astype(np.uint8)
        rewards[-1] = 1
        print(len(actions), len(agentview_images))
        assert len(actions) == len(agentview_images)
        print(len(actions))

        ep_data_grp = grp.create_group(f"demo_{i}")

        obs_grp = ep_data_grp.create_group("obs")
        if not args.no_proprio:
            obs_grp.create_dataset(
                "gripper_states", data=np.stack(gripper_states, axis=0)
            )
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

        obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
        obs_grp.create_dataset(
            "eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0)
        )
        if args.use_depth:
            obs_grp.create_dataset(
                "agentview_depth", data=np.stack(agentview_depths, axis=0)
            )
            obs_grp.create_dataset(
                "eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0)
            )

        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("states", data=states)
        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)
        ep_data_grp.attrs["num_samples"] = len(agentview_images)
        ep_data_grp.attrs["model_file"] = model_xml
        ep_data_grp.attrs["init_state"] = states[init_idx]
        total_len += len(agentview_images)

    grp.attrs["num_demos"] = len(demos)
    grp.attrs["total"] = total_len
    env.close()

    h5py_f.close()
    f.close()

    print("The created dataset is saved in the following path: ")
    print(hdf5_path)


if __name__ == "__main__":
    main()

def is_empty_input_spacemouse(action):
    if np.all(action[:6] == 0) and action[6] == -1 and np.all(action[7:11] == 0):
        return True
    return False

def collect_human_trajectory(
    env, device, arm, env_configuration, mirror_actions,
    render=True, max_fr=None,
    print_info=True,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    ep_meta = env.get_ep_meta()
    # print(json.dumps(ep_meta, indent=4))
    lang = ep_meta.get("lang", None)
    if print_info and lang is not None:
        print(colored(f"Instruction: {lang}", "green"))

    # degugging: code block here to quickly test and close env
    # env.close()
    # return None, True

    if render:
        # ID = 2 always corresponds to agentview
        env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    nonzero_ac_seen = False

    # Set active robot
    active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

    if active_robot.is_mobile:
        zero_action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
    else:
        zero_action = np.array([0, 0, 0, 0, 0, 0, -1])
    for _ in range(1):
        # do a dummy step thru base env to initalize things, but don't record the step
        if isinstance(env, DataCollectionWrapper):
            env.env.step(zero_action)
        else:
            env.step(zero_action)

    discard_traj = False

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Get the newest action
        input_action, _ = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
            mirror_actions=mirror_actions,
        )

        # If action is none, then this a reset so we should break
        if input_action is None:
            discard_traj = True
            break

        if is_empty_input_spacemouse(input_action):
            if not nonzero_ac_seen:
                if render:
                    env.render()
                continue
        else:
            nonzero_ac_seen = True

        if env.robots[0].is_mobile:
            arm_actions = input_action[:6]
            # arm_actions = np.concatenate([arm_actions, ])

            # flip some actions
            arm_actions[0], arm_actions[1] = arm_actions[1], -arm_actions[0]
            arm_actions[3], arm_actions[4] = arm_actions[4], -arm_actions[3]

            base_action = input_action[7:10]
            torso_action = input_action[10:11]

            if np.abs(torso_action[0]) < 0.50:
                torso_action[:] = 0.0

            # flip some actions
            base_action[0], base_action[1] = base_action[1], -base_action[0]

            action = np.concatenate((
                arm_actions,
                np.repeat(input_action[6:7], env.robots[0].gripper[arm].dof),
                base_action,
                torso_action,
            ))
            mode_action = input_action[-1]

            env.robots[0].enable_parts(base=True, right=True, left=True, torso=True)
            if mode_action > 0:
                action = np.concatenate((action, [1]))
            else:
                action = np.concatenate((action, [-1]))
        else:
            arm_actions = input_action
            action = env.robots[0].create_action_vector(
                {
                    arm: arm_actions[:-1], 
                    f"{arm}_gripper": arm_actions[-1:]
                }
            )

        # Run environment step
        obs, _, _, _ = env.step(action)
        if render:
            env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # with open("/home/soroushn/tmp/model.xml", "w") as f:
        #     f.write(env.model.get_xml())
        # exit()

    if nonzero_ac_seen and hasattr(env, "ep_directory"):
        ep_directory = env.ep_directory
    else:
        ep_directory = None
        
    # cleanup for end of data collection episodes
    env.close()

    return ep_directory, discard_traj


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        # success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])
            # success = success or dic["successful"]

        if len(states) == 0:
            continue

        # # Add only the successful demonstration to dataset
        # if success:
        
        # print("Demonstration is successful and has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))
        
        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} successful demos so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument("--robots", nargs="+", type=str, default="PandaMobile", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument(
        "--obj_groups", type=str, nargs='+', default=None, 
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file"
    )

    parser.add_argument("--camera", type=str, default=None, help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="spacemouse", choices=["keyboard", "keyboardmobile", "spacemouse", "dummy"])
    parser.add_argument("--pos-sensitivity", type=float, default=4.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=4.0, help="How much to scale rotation user inputs")
    
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"])
    parser.add_argument("--max_fr", default=30, type=int, help="If specified, limit the frame rate")

    parser.add_argument("--layout", type=int, nargs='+', default=-1)
    parser.add_argument("--style", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    parser.add_argument("--generative_textures", action="store_true")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    env_name = args.environment

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if args.generative_textures is True:
        config["generative_textures"] = "100p"

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]: # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print("Specifying 'obj_groups' in non-kitchen environment does not have an effect.")
        mirror_actions = False
        if args.camera is None:
            args.camera = "agentview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None
    else:
        mirror_actions = True
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})
        if args.camera is None:
            args.camera = "robot0_frontview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None

        config["translucent_robot"] = True

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

    # Create environment
    env = suite.make(
        **config,
        has_renderer=(args.renderer != "mjviewer"),
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y-%m-%d-%H-%M-%S')

    if not args.debug:
        # wrap the environment with data collection wrapper
        tmp_directory = "/tmp/{}".format(time_str)
        env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    else:
        raise ValueError

    # make a new timestamped directory
    new_dir = os.path.join(args.directory, time_str)
    os.makedirs(new_dir)

    excluded_eps = []

    # collect demonstrations
    while True:
        print()
        ep_directory, discard_traj = collect_human_trajectory(
            env, device, args.arm, args.config, mirror_actions, render=(args.renderer != "mjviewer"),
            max_fr=args.max_fr,
        )

        print("Keep traj?", not discard_traj)
        
        if not args.debug:
            if discard_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split('/')[-1])
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps)
