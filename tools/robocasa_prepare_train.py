import os, io, sys
import json
import numpy as np
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '~/yanfeng/project/robotic/Metaworld/mujoco210'
sys.path.append('~/yanfeng/project/robotic/RoboUniview')
from robouniview.data.zipreader import ZipReader
import zipfile, pickle

root = '~/yanfeng/data/robotics/robocasa_uniview'



meta_tasks = [name for name in os.listdir(root)]
train_metadata = []
actions=[]

for task in meta_tasks:
    if not os.path.isdir(os.path.join(root, task)): continue
    meta_eposides = [name for name in os.listdir(os.path.join(root, task)) if '.zip' in name]
    print(meta_eposides)
    for eposide in meta_eposides:

        try:
            zip_file = os.path.join(task, eposide)
            frames = pickle.load(io.BytesIO(ZipReader.read(os.path.join(root, zip_file), 'param/param.pickle')))
            video_len = len(frames)
            env_param=np.load(io.BytesIO(ZipReader.read(os.path.join(root, zip_file), 'param/env.npy')), allow_pickle=True).item()
            lang = json.loads(str(env_param['initial_state']["ep_meta"])).get("lang", None)
            
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

            train_metadata.append([zip_file, video_len, task, lang, base_extrinsic.tolist()])

            for f in frames:
                actions.append(f['action'])
        except Exception as e:
            print(e)
            continue

with open(os.path.join(root, "train_metaworld.json"), 'w') as json_file:
    json.dump(train_metadata, json_file)

actions=np.array(actions)
print(actions.min(0), actions.max(0))
