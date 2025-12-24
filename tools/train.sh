
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
conda activate ~/yanfeng/software/anaconda3/envs/RBMM214/

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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/software/anaconda3/envs/RBMM214/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/dri/swrast_dri.so
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.8
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/yanfeng/project/robotic/Metaworld/mujoco210/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/zhongyufeng/code/robot/robo_mm_new
export VK_ICD_FILENAMES=~/yanfeng/.vulkan/icd.d/nvidia_icd.json



PY_ARGS=${@:2} # 第2个输入参数后边的值

# 脚本运行失败，报错
set -o pipefail
#sed -e  ：直接在指令列模式上進行 sed 的動作編輯；
OUTPUT_BASE=$(echo $2 | sed -e "s/config/exps/g" | sed -e "s/.yaml$//g")
OUTPUT_BASE=~/zhongyufeng/code/robot/robo_mm_new/$OUTPUT_BASE
mkdir -p $OUTPUT_BASE

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

if ((NODE_RANK == 0)); then
  for RUN in $(seq 100); do
    ls $OUTPUT_BASE | grep run$RUN && continue
    OUTPUT_DIR=$OUTPUT_BASE/run$RUN
    mkdir $OUTPUT_DIR && break
  done

  # clean up *.pyc files
  rmpyc() {
    rm -rf $(find -name __pycache__)
    rm -rf $(find -name "*.pyc")
  }

  # run backup
  echo "Backing up to log dir: $OUTPUT_DIR"
  rmpyc && cp -r robouniview $2 $OUTPUT_DIR
  echo " ...Done"

  # tar src to avoid future editing
  cleanup() {
    echo "Packing source code"
    rmpyc
    # tar -zcf models datasets util main.py engine.py eval.py submit.py --remove-files
    echo " ...Done"
  }

  # pushd $OUTPUT_DIR
  log git status
  echo "Logging git status"
  git status > $OUTPUT_DIR/git_status
  git rev-parse HEAD > $OUTPUT_DIR/git_tag
  git diff > $OUTPUT_DIR/git_diff

else
  # 3 minutes
  sleep 180
  for RUN in $(seq 100); do
    ls $OUTPUT_BASE | grep run$RUN && continue
    let "ITERRUN=$RUN-1"
    OUTPUT_DIR=$OUTPUT_BASE/run$ITERRUN
    break
  done
fi

trap cleanup EXIT
echo $PY_ARGS > desc
echo " ...Done"

args=$2
echo $args
GPUS_NUM=$1
# NODE_RANK=0
# MASTER_ADDR=localhost
# NODE_NUM=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1 
# export NCCL_DEBUG=INFO

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2

# export NCCL_DEBUG=INFO
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=1
# export NCCL_P2P_DISABLE=1

# 例如在使用RoCE时
# export NCCL_IB_HCA=mlx5_0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0   # 或者其他正确的网络接口名称
# export UCX_WARN_UNUSED_ENV_VARS=n
# export UCX_MAX_RNDV_RAILS=1
# export UCX_TCP_MAX_CONN_RETRIES=5

# export CUDA_LAUNCH_BLOCKING=1

# 设置环境变量
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_IB_GID_INDEX=-1
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_NET_GDR_READ=0
# export NCCL_SHM_DISABLE=0

# 自动检测和配置 NCCL 环境变量的函数
link_layer=$(cat /sys/class/infiniband/mlx5_6/ports/1/link_layer)
if [[ "${link_layer}" == "Ethernet" ]]; then
   # A100/H800-RCoE
   source /workdir/export_gid_index.sh
   export NCCL_NVLS_ENABLE=0
elif [[ "${link_layer}" == "InfiniBand" ]]; then
   # H800-IB
   export NCCL_IB_HCA=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
   export LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   export NCCL_COLLNET_ENABLE=1
   export USE_SHARP=1
   export NCCL_NVLS_ENABLE=0
   unset NCCL_IB_GID_INDEX
fi
export NCCL_DEBUG=TRACE 
ulimit -l


# 打印环境变量以确认
echo "NCCL_IB_DISABLE is set to $NCCL_IB_DISABLE"
echo "NCCL_SOCKET_IFNAME is set to $NCCL_SOCKET_IFNAME"
echo "NCCL_IB_HCA is set to $NCCL_IB_HCA"
echo "NCCL_IB_GID_INDEX is set to $NCCL_IB_GID_INDEX"
printenv | grep NCCL

# # 设置OpenMP线程数
# export OMP_NUM_THREADS=196
# # 设置MKL线程数
# export MKL_NUM_THREADS=196
# # 设置NumPy线程数
# export NUMEXPR_NUM_THREADS=196
# # 设置PyTorch线程数
# export TORCH_NUM_THREADS=196

# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# torchrun --nnodes=${NODE_NUM} --nproc_per_node=8 --master_port=6042 robot_flamingo/train/train_calvin.py  ${args} --run_name $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log
# python -m torch.distributed.launch --nproc_per_node=${GPUS_NUM} --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port 29502 --use_env robouniview/train/train.py --config ${args} --save_dir $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log
python -m torch.distributed.launch --nproc_per_node=${GPUS_NUM} --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port 29502 --use_env robouniview/train/train.py --config ${args} |& tee -a $OUTPUT_DIR/output.log
# torchrun --nproc_per_node=${GPUS_NUM} --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port 29502 robouniview/train/train.py --config ${args} --save_dir $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log

# torchrun --nnodes=${NODE_NUM} --nproc_per_node=${GPUS_NUM} --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --rdzv_backend=c10d  robouniview/train/train.py --config ${args} --save_dir $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log

# torchrun --nproc_per_node=${GPUS_NUM} --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}  robouniview/train/train.py --config ${args} --save_dir $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log

python ~/zhongyufeng/code/robot/robo_mm_nav/robo_mm_new/occ.py