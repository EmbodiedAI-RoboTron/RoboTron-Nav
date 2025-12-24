#!/usr/bin/env bash

conda activate robotron-nav

sudo apt --fix-broken install
export COPPELIASIM_ROOT=~/yanfeng/project/robotic/RLBench/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export QT_DEBUG_PLUGINS=1

sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

sudo apt-get update -y
sudo apt-get install -y  libegl1-mesa libegl1-mesa-dev

sudo apt install -y mesa-utils libosmesa6-dev llvm
sudo apt-get -y install meson
sudo apt-get -y build-dep mesa

sudo apt-get -y install freeglut3
sudo apt-get -y install freeglut3-dev

sudo apt-get install -y libgl1-mesa-dri

sudo apt update -y
sudo apt install -y xvfb
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99

sudo apt-get install -y libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev libqt5svg5-dev
# echo "keyboard-configuration  keyboard-configuration/layout select English (US)" | sudo debconf-set-selections
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y xorg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/dri
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/software/anaconda3/envs/RBMM214/lib

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/dri/swrast_dri.so 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.8
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/project/robotic/Metaworld/mujoco210/bin
sudo apt-get install patchelf -y
sudo apt-get install libc6-dev -y

export VK_ICD_FILENAMES=~/yanfeng/.vulkan/icd.d/nvidia_icd.json

sudo apt-get install imagemagick -y
sudo apt install vulkan-tools

export MASTER_ADDR=localhost
export MASTER_PORT=5679
export PROCESS_SPAWN_TIMEOUT=30

CKPT=$3
RESUME=$2
GPU_NUMS=$1

torchrun --nproc_per_node ${GPU_NUMS} --nnodes 1 --node_rank 0 --master_addr=localhost --master_port 29502 robouniview/eval/eval.py --evaluate_from_checkpoint ${CKPT} --resume ${RESUME}