# benchmark_naive_ddp.py
# -------------------------------------------------------------
# CSE 599O: Distributed Training Basics
#
# Implement a naive DDP version that reproduces the same model
# state as single-process training.
#
# The TA will test your implementation with the following commands:
#
# 1. To verify that DDP matches baseline (toy model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model toy
# Expected output: "Naive DDP matches baseline!"
#
# 2. To output communication and step time (transformer model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model transformer
# Expected output: communication and step time statistics
#
# -------------------------------------------------------------

# Any necessary imports can be added here.
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from cse599o_basics.optimizer import AdamW
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.utils import cross_entropy_loss
from tests.common import ToyModel

# Any necessary helper functions can be defined here.
# add near the top
def reset_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Initialize NCCL for GPU-based distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# You can change the function and variable names as needed.
def run_naive_ddp_worker(rank, world_size, data, target, num_steps, result_queue):
    """Run one DDP worker process."""
    setup_ddp(rank, world_size)
    try:
        reset_seed(42)       
        print(f"Rank {rank} starting naive DDP worker...")
        # Split data among ranks
        global_batch_size = data.size(0)
        local_data = data[rank:global_batch_size:world_size].to(f"cuda:{rank}")
        local_target = target[rank:global_batch_size:world_size].to(f"cuda:{rank}")

        model = ToyModel().to(f"cuda:{rank}")
        # use broadcast to share initial model parameters from rank 0 to all other ranks
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        # print(f"Rank {rank} model parameters after broadcast:")
        # for name, param in model.named_parameters():
        #     print(f"  {name}: {param.data}")

        optimizer = AdamW(model.parameters(), lr=0.001)

        model.train()
        batch_size = local_data.size(0) // num_steps
        print(f"Rank {rank} batch size: {batch_size}")
        for step in range(num_steps):
            indices = torch.arange(step * batch_size, (step + 1) * batch_size)
            print(f"Rank {rank}, Step {step + 1}, processing indices: {indices.tolist()}")
            local_batch_data = local_data[indices]
            local_batch_target = local_target[indices]
            optimizer.zero_grad()
            output = model(local_batch_data)
            loss = cross_entropy_loss(output, local_batch_target)
            loss.backward()
            print(f"Rank {rank}, Step {step + 1}, Loss before all-reduce: {loss.item():.4f}")
            # dist.all_reduce(loss.grad)  # Naive gradient synchronization
            for p in model.parameters():
                if p.grad is None:
                    continue   # param unused this step; skip
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optimizer.step()


        if rank == 0:
            # TODO: Collect and return the model state from rank 0
            # model_state = {name: param.data for name, param in model.named_parameters()}
            # result_queue.put(model_state)
            state = model.state_dict()
            cpu_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            result_queue.put(cpu_state)


    finally:
        cleanup_ddp()

# You can change the function and variable names as needed.
def run_baseline(data, target, num_steps):
    """Run single-process baseline for comparison."""
    reset_seed(42)       
    model = ToyModel().to("cuda:0")
    optimizer = AdamW(model.parameters(), lr=0.001)
    model.train()
    batch_size = data.size(0) // num_steps
    for step in range(num_steps):
        indices = torch.arange(step * batch_size, (step + 1) * batch_size)
        batch_data = data[indices].to("cuda:0")
        batch_target = target[indices].to("cuda:0")
        optimizer.zero_grad()
        output = model(batch_data)
        loss = cross_entropy_loss(output, batch_target)
        loss.backward()
        optimizer.step()
        print(f"Baseline, Step {step + 1}, Loss: {loss.item():.4f}")
        
    
    state = model.state_dict()
    cpu_state = {k: v.detach().cpu().clone() for k, v in state.items()}

    return cpu_state

# You can change the function and variable names as needed.
def verify_naive_ddp():
    """Benchmark and verify naive DDP."""
    world_size = 2
    num_steps = 5
    data = torch.randn(10, 10)
    target = torch.randint(0,5, (10,))

    # Run baseline
    print("Starting baseline training...")
    no_ddp_state = run_baseline(data, target, num_steps)

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    print("Starting naive DDP training...")
    mp.spawn(
        run_naive_ddp_worker,
        args=(world_size, data, target, num_steps, result_queue),
        nprocs=world_size,
        join=True,
    )

    # Get model state from DDP
    ddp_state = result_queue.get()
    
    # print("Verifying model states...")
    # print("Baseline model state:")
    # for name, param in no_ddp_state.items():
    #     print(f"  {name}: {param}")
    # print("DDP model state:")
    # for name, param in ddp_state.items():
    #     print(f"  {name}: {param}")

    assert len(no_ddp_state) > 0, "model state from baseline is empty"
    print("Comparing model states...")
    for name in no_ddp_state:
        assert torch.allclose(no_ddp_state[name], ddp_state[name], atol=1e-6)
    print("Naive DDP matches baseline!")
  
# You can change the function and variable names as needed.  
def timing_naive_ddp():
    """Timing benchmark for naive DDP with transformer model."""
    # TODO
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["toy", "transformer"], default="toy")
    args = parser.parse_args()

    if args.model == "toy":
        verify_naive_ddp()
    elif args.model == "transformer":
        timing_naive_ddp()