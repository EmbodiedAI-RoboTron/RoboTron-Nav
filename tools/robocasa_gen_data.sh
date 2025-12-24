
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
conda activate ~/yanfeng/software/anaconda3/envs/RF/

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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/software/anaconda3/envs/RF/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/dri/swrast_dri.so 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.8
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/project/robotic/Metaworld/mujoco210/bin
sudo apt-get install patchelf -y
sudo apt-get install libc6-dev -y



python tools/robocasa_playback_dataset.py
# python data_generate/metaworld_data.py
