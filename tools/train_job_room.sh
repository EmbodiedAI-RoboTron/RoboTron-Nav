source run.sh

torchrun --nproc_per_node=8 --nnodes 1 --node_rank 0 --master_addr=localhost --master_port 29502 robouniview/train/train.py --config config/uvf_chores_room.yaml