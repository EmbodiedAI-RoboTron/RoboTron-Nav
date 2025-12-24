import os, io
import json
import numpy as np
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '~/yanfeng/project/robotic/Metaworld/mujoco210'

from robouniview.data.zipreader import ZipReader
import zipfile, pickle

root = '~/yanfeng/data/robotics/metaworld_uniview/'


all_v2_pol_instance = {'assembly-v2': ['assemble nut', 'Pick up a nut and place it onto a peg'],
                       'basketball-v2': ['basketball', 'Dunk the basketball into the basket'],
                       'bin-picking-v2': ['pick bin', 'Grasp the puck from one bin and place it into another bin'],
                       'box-close-v2': ['close box', 'Grasp the cover and close the box with it'],
                       'button-press-topdown-v2': ['press button top', 'Press a button from the top'],
                       'button-press-topdown-wall-v2': ['press button top wall',
                                                        'Bypass a wall and press a button from the top'],
                       'button-press-v2': ['press button', 'Press a button'],
                       'button-press-wall-v2': ['press button wall', 'Bypass a wall and press a button'],
                       'coffee-button-v2': ['get coffee', 'Push a button on the coffee machine'],
                       'coffee-pull-v2': ['pull mug', 'Pull a mug from a coffee machine'],
                       'coffee-push-v2': ['push mug', 'Push a mug under a coffee machine'],
                       'dial-turn-v2': ['turn dial', 'Rotate a dial 180 degrees'],
                       'disassemble-v2': ['disassemble nut', 'pick a nut out of the a peg'],
                       'door-close-v2': ['close door', 'Close a door with a revolving joint'],
                       'door-lock-v2': ['lock door', 'Lock the door by rotating the lock clockwise'],
                       'door-open-v2': ['open door', 'Open a door with a revolving joint'],
                       'door-unlock-v2': ['unlock door', 'Unlock the door by rotating the lock counter-clockwise'],
                       'hand-insert-v2': ['insert hand', 'Insert the gripper into a hole'],
                       'drawer-close-v2': ['close drawer', 'Push and close a drawer'],
                       'drawer-open-v2': ['open drawer', 'Open a drawer'],
                       'faucet-open-v2': ['turn on faucet', 'Rotate the faucet counter-clockwise'],
                       'faucet-close-v2': ['Turn off faucet', 'Rotate the faucet clockwise'],
                       'hammer-v2': ['hammer', 'Hammer a screw on the wall'],
                       'handle-press-side-v2': ['press handle side', 'Press a handle down sideways'],
                       'handle-press-v2': ['press handle', 'Press a handle down'],
                       'handle-pull-side-v2': ['pull handle side', 'Pull a handle up sideways'],
                       'handle-pull-v2': ['pull handle', 'Pull a handle up'],
                       'lever-pull-v2': ['Pull lever', 'Pull a lever down 90 degrees'],
                       'peg-insert-side-v2': ['insert peg side', 'Insert a peg sideways'],
                       'pick-place-wall-v2': ['pick and place wall', 'Pick a puck, bypass a wall and place the puck'],
                       'pick-out-of-hole-v2': ['pick out of hole', 'Pick up a puck from a hole'],
                       'reach-v2': ['reach', 'reach a goal position'],
                       'push-back-v2': ['push back', 'push back'],
                       'push-v2': ['Push', 'Push the puck to a goal'],
                       'pick-place-v2': ['pick and place', 'Pick and place a puck to a goal'],
                       'plate-slide-v2': ['slide plate', 'Slide a plate into a cabinet'],
                       'plate-slide-side-v2': ['slide plate side', 'Slide a plate into a cabinet sideways'],
                       'plate-slide-back-v2': ['retrieve plate', 'Get a plate from the cabinet'],
                       'plate-slide-back-side-v2': ['retrieve plate side', 'Get a plate from the cabinet sideways'],
                       'peg-unplug-side-v2': ['unplug peg', 'Unplug a peg sideways'],
                       'soccer-v2': ['soccer', 'Kick a soccer into the goal'],
                       'stick-push-v2': ['push with stick', 'Grasp a stick and push a box using the stick'],
                       'stick-pull-v2': ['pull with stick', 'Grasp a stick and pull a box with the stick'],
                       'push-wall-v2': ['push with wall', 'Bypass a wall and push a puck to a goal'],
                       'reach-wall-v2': ['reach with wall', 'Bypass a wall and reach a goal'],
                       'shelf-place-v2': ['place onto shelf', 'pick and place a puck onto a shelf'],
                       'sweep-into-v2': ['sweep into hole', 'Sweep a puck into a hole'],
                       'sweep-v2': ['Sweep', 'Sweep a puck off the table'],
                       'window-open-v2': ['open window', 'Push and open a window'],
                       'window-close-v2': ['close window', 'Push and close a window']}


meta_tasks = [name for name in os.listdir(root) if 'v2' in name]
train_metadata = []

actions=[]
for task in meta_tasks:

    meta_eposides = [name for name in os.listdir(os.path.join(root, task)) if '.zip' in name]
    print(meta_eposides)
    for eposide in meta_eposides:

        try:
            zip_file = os.path.join(task, eposide)
            frames = pickle.load(io.BytesIO(ZipReader.read(os.path.join(root, zip_file), 'param/param.pickle')))
            
            video_len = len(frames)
            task_desc = all_v2_pol_instance[task.split('v2')[0]+'v2'][1]
        
            train_metadata.append([zip_file, video_len, task, task_desc])

            for f in frames:
                actions.append(f['action'])
        except:
            continue

with open(os.path.join(root, "train_metaworld.json"), 'w') as json_file:
    json.dump(train_metadata, json_file)

actions=np.array(actions)
print(actions.min(0), actions.max(0))
