""" Main training script """

import argparse
import glob
import os
import random
# from robot_flamingo.eval.eval_utils import eval_one_epoch_calvin_ddp
from torch.distributed.elastic.multiprocessing.errors import record

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import torch
# import wandb
from torch.nn.parallel import DistributedDataParallel as DDP


from PIL import Image

import socket
import struct
import pickle
import time


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def recv_data(so):
    result_len = struct.unpack('!I', so.recv(4))[0]
    data = b''
    while len(data) < result_len:
        packet = so.recv(min(4096, result_len - len(data)))
        if not packet:
            break
        data += packet
    return data


def send_data(so, data):
    so.sendall(struct.pack('!I', len(data)))
    so.sendall(data)

if 1:
    local_ip_address, local_ip_port = "10.166.174.69", 8415
    ip_port = (local_ip_address, local_ip_port)
    so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    so.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    so.bind(ip_port)
    so.listen(2)
    while True:
        print('waitting for connection')
        con, address = so.accept()
        print('connected:', address)
        
        planned_actions = []
        try:
            
            while True:
                req = recv_data(con)
                
                print(req)

                # data = pickle.loads(req)
                # obs = data['obs']
                # lang_annotation = data['lang_annotation']

                # # debug
                # time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
                # img = np.concatenate([obs['rgb_obs']['rgb_gripper'], obs['rgb_obs']['rgb_static']], 0)
                # img = Image.fromarray(img)
                # img.save('debug/%s.png' % time_str)

                # action = model.step(obs, lang_annotation, (len(planned_actions) == 0))
                # if len(planned_actions) == 0:
                #     if action.shape == (7,):
                #         planned_actions.append(action)
                #     else:
                #         planned_actions.extend([action[i] for i in range(action.shape[0])])
                # action = planned_actions.pop(0)
                # if model.use_diff:
                #     model.action_hist_queue.append(action)
                # action[:6] /= np.array([20, 20, 20, 10, 10, 6])
    
                # print(action)
                res = pickle.dumps(req)
                send_data(con, res)
        except Exception as e:
            print(e)
        finally:
            con.close()
        print('on trajectory finished')


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = '1'
    main()
