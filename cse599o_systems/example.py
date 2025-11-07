import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Initialize NCCL for GPU-based distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    # Create data tensor on this rank’s GPU
    data = torch.randint(0, 10, (3,), device=f"cuda:{rank}")
    print(f"Rank {rank} data (before all-reduce): {data}")
    # Perform all-reduce across GPUs
    # Measure the runtime of all-reduce operation in all ranks
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    dist.all_reduce(data)
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)  # milliseconds
    print(f"Rank {rank} data (after all-reduce): {data}")
    print(f"Rank {rank} all-reduce elapsed time: {elapsed_time} ms")
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() # auto-detect GPUs
    if world_size < 2:
        raise RuntimeError("Need at least 2 GPUs to run this example.")
    mp.spawn(distributed_demo, args=(world_size,), nprocs=world_size, join=True)