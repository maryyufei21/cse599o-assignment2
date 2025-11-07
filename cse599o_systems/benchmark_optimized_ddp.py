# benchmark_optimized_ddp.py
# -------------------------------------------------------------
# CSE 599O
#
# Extend your DDP benchmark to evaluate three optimized variants
# for the Transformer model:
#   (1) run_flat       
#   (2) run_individual 
#   (3) run_bucketed   
#
# The TA will execute your script using commands like:
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode flat
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode individual
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode bucketed --bucket-mb 10
#
# Each function should measure and print out the following statistics:
#   - iteration time per step  → append to iteration_times
#   - communication time per step → append to comm_times
# -------------------------------------------------------------

import argparse
import torch
import os
import torch.distributed as dist
# Any other necessary imports can be added here.

# Any necessary helper functions can be defined here.

# You can change the function and variable names as needed.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# ============================================================
# (0) Naive DDP
# ============================================================
# You can change the function and variable names as needed.
def run_naive(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """A naive DDP training loop for reference."""
    # TODO:
    pass

# ============================================================
# (1) Flat DDP
# ============================================================
# You can change the function and variable names as needed.
def run_flat(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce a single flattened gradient tensor."""
    # TODO:
    pass


# ============================================================
# (2) Individual DDP
# ============================================================
# You can change the function and variable names as needed.
def run_individual(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce each parameter's gradient individually."""
    # TODO:
    pass


# ============================================================
# (3) Bucketed DDP
# ============================================================
# You can change the function and variable names as needed.
def run_bucketed(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, bucket_mb):
    """Group gradients into buckets and all-reduce each bucket."""
    # TODO:
    pass


# ============================================================
# Benchmark Function
# ============================================================
# You can change the function and variable names as needed.
def benchmark_optimized_ddp():
    """Benchmark DDP variants on the Transformer model."""
    parser = argparse.ArgumentParser(description="Benchmark optimized DDP variants.")
    parser.add_argument(
        "--mode",
        type=str,
        default="flat",
        choices=["flat", "individual", "bucketed"],
        help="Select which DDP variant to benchmark.",
    )
    parser.add_argument(
        "--bucket-mb",
        type=int,
        default=10,
        help="Bucket size (in MB) for the bucketed DDP variant.",
    )
    args = parser.parse_args()

    # Example placeholders
    num_iters, num_warmup = 5, 2
    iteration_times, comm_times = [], []
    
    # DDP setup
    # TODO: Initialize distributed process group

    # Construct model and move to GPU
    # TODO: Define model parameters

    # Construct optimizer
    # TODO: Define optimizer
    
    # Dummy data
    # TODO: Create input data

    if args.mode == "naive":
        run_naive(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "flat":
        run_flat(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "individual":
        run_individual(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "bucketed":
        run_bucketed(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, args.bucket_mb)

    print(f"Mode: {args.mode}")
    print(f"Iteration times: {iteration_times}")
    print(f"Communication times: {comm_times}")


if __name__ == "__main__":
    benchmark_optimized_ddp()
