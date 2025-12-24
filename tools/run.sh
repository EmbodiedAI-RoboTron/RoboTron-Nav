source ~/common/.condasetupdolphinfs
conda activate ~/qiaolm/conda/envs/internvl-cu117
echo "env activated"

cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
echo "worker list command is $worker_list_command"
eval worker_list=`python -c "$worker_list_command"`
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
dist_url="tcp://$master_addr:$master_port"
index_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
eval NODE_RANK=`python -c "$index_command"`
echo "node rank is $NODE_RANK"
MASTER_NODE=${worker_strs[0]}
echo "master node is $MASTER_NODE"

# 定义公共路径
MVR_COMMON_DIR=~/mvrcommon
JOB_DIR_POLLING=${MVR_COMMON_DIR}/.sharedtasks/$1
MISC_DIR=${MVR_COMMON_DIR}/.sharedtasks/.misc
FLAG_FILEDIR=${MISC_DIR}/JOB-FINISHED-FLAG
FLAG_FILEPATH=$FLAG_FILEDIR/$MASTER_NODE
FLAG_FILEPATH_NODE_READY=$FLAG_FILEDIR/${worker_strs[$NODE_RANK]}
mkdir -p $JOB_DIR_POLLING
mkdir -p $FLAG_FILEDIR

# 开始父任务
PID=-1
PRE_TASK=none
cd $JOB_DIR_POLLING
while true
do
    CUR_TASK=$(ls ${JOB_DIR_POLLING} | grep 'job_.*sh' | sort | tail -n 1)  # JOB_DIR_POLLING目录下子任务必须以job_开头
    echo $CUR_TASK
    if [[ "$PRE_TASK" == "$CUR_TASK" ]]; then
        date && echo "" # do nothing
        sleep 120
        # NOTE. that if CUR_TASK finished, this script will be idle
        # TODO. if idle for 1 hour, break
    else
        echo "✅ Found New Task ${CUR_TASK}"
        sleep 60
        if [[ "$PRE_TASK" == "none" ]]; then
            echo "PASS"
        else
            # kill PRE_TASK and all its child processes if not none
            pstree -p $PID | grep -o "([[:digit:]]*)" | grep -o  "[[:digit:]]*" | xargs kill
            ps aux | grep python | awk '{ print $2 }'| xargs kill   # deepspeed is needed
            # clean flag files
            if (($NODE_RANK == 0))
            then
                rm -f $FLAG_FILEPATH
            else
                rm -f $FLAG_FILEPATH_NODE_READY
            fi
            for i in {1..10}
            do
                echo "🧹 Cleaned $PRE_TASK"
            done
            sleep 60 # sleep for a while, let the processes clean up
        fi
        PRE_TASK=$CUR_TASK
        cat $CUR_TASK && bash $CUR_TASK run &
        PID=$!
        echo $PID
    fi
    sleep 60
done
