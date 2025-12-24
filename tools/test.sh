
#!/usr/bin/env bash
###
 # @Author: 颜峰 && bphengyan@163.com
 # @Date: 2023-05-19 17:19:11
 # @LastEditors: 颜峰 && bphengyan@163.com
 # @LastEditTime: 2023-05-22 09:54:42
 # @FilePath: /CO-MOT/tools/train.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
### 
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

# 打印所有指令
set -x

eval "$('~/yanfeng/software/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate ~/yanfeng/software/anaconda3/envs/RBMM/

echo abc123 | sudo -S apt-get install openssh-server -y 

echo abc123 | sudo -S apt --fix-broken install -y

echo abc123 | sudo -S  apt-get -y install libegl1-mesa libegl1
echo abc123 | sudo -S  apt-get -y install libgl1

echo abc123 | sudo -S  apt-get update -y 
echo abc123 | sudo -S  apt-get install -y  libegl1-mesa libegl1-mesa-dev

echo abc123 | sudo -S  apt install -y mesa-utils libosmesa6-dev llvm
echo abc123 | sudo -S  apt-get -y install meson
echo abc123 | sudo -S  apt-get -y build-dep mesa

echo abc123 | sudo -S  apt-get -y install freeglut3
echo abc123 | sudo -S  apt-get -y install freeglut3-dev

echo abc123 | sudo -S  apt-get install -y libgl1-mesa-dri

echo abc123 | sudo -S  apt update -y 
echo abc123 | sudo -S  apt install -y xvfb
echo abc123 | sudo -S  apt-get install patchelf -y
echo abc123 | sudo -S  apt-get install libc6-dev -y
echo abc123 | sudo -S  apt-get install -y libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev libqt5svg5-dev
# echo "keyboard-configuration  keyboard-configuration/layout select English (US)" | sudo debconf-set-selections
echo abc123 | sudo -S  DEBIAN_FRONTEND=noninteractive apt-get install -y xorg

echo abc123 | sudo -S  apt-get install git -y 

Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99

export COPPELIASIM_ROOT=~/yanfeng/project/robotic/RLBench/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export QT_DEBUG_PLUGINS=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/dri
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/software/anaconda3/envs/RF/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/dri/swrast_dri.so 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.8
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/project/robotic/Metaworld/mujoco210/bin

export VK_ICD_FILENAMES=~/yanfeng/.vulkan/icd.d/nvidia_icd.json

cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import tools.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
echo "worker list command is $worker_list_command"
eval worker_list=`python -c "$worker_list_command"`
echo "worker list is $worker_list"
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
index_command="import tools.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
eval node_rank=`python -c "$index_command"`
echo "node rank is $node_rank"
dist_url="tcp://$master_addr:$master_port"
echo "dist url is $dist_url"
PYTHONPATH=$PYTHONPATH:../ \
# python tools/run_net.py \
#    --num_shards 8 \
#    --shard_id $node_rank \
#    --dist_url $dist_url \
#    --cfg configs/verb/MVIT_B_32x2_CONV.yaml

MASTER_ADDR=${MASTER_ADDR:-$master_addr}
MASTER_PORT=${MASTER_PORT:-$master_port}
NODE_RANK=${NODE_RANK:-$node_rank}
# let "NNODES=GPUS/GPUS_PER_NODE"

NODE_NUM=${#worker_strs[@]}  
echo "node num is $NODE_NUM"


args=$2
echo $args
GPUS_NUM=$1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1 
export NCCL_DEBUG=INFO

# 保存当前 IFS 的值
OLD_IFS=$IFS
# 设置 IFS 为 .
IFS='.'
# 使用 read 命令将字符串拆分成数组
read -ra parts <<< "$args"
# 恢复原 IFS 值
IFS=$OLD_IFS
OUTPUT_DIR=${parts[0]}
mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=${GPUS_NUM} --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port 29502 robouniview/eval/eval.py --evaluate_from_checkpoint ${args}  |& tee -a $OUTPUT_DIR/output.log