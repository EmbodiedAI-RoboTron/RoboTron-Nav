import torch.distributed as dist
import torch.multiprocessing as mp

def gather_example(rank, world_size):
    # 初始化分布式进程组
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # 各进程的对象 (不同进程发送不同大小的对象)
    object_to_send = list(range(rank * 3))  # 例如，rank 0 发送空列表 [], rank 1 发送 [0, 1, 2], rank 2 发送 [0, 1, 2, 3, 4, 5]
    print(object_to_send)
    # 仅rank 0进程需要有空间来接收对象
    if rank == 0:
        gathered_objects = [None for _ in range(world_size)]
    else:
        gathered_objects = None

    # 收集对象
    dist.gather_object(object_to_send, gathered_objects, dst=0)

    if rank == 0:
        print(f"Gathered objects: {gathered_objects}")

    # 销毁分布式进程组
    dist.destroy_process_group()

def run_example():
    world_size = 4
    mp.spawn(gather_example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_example()