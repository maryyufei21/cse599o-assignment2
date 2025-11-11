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
import os, json
import torch.distributed as dist
import torch.multiprocessing as mp
# Any other necessary imports can be added here.
from benchmark_naive_ddp import reset_seed
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.optimizer import AdamW
from cse599o_systems.ddp_overlap_individual_parameters import DDPOverlapIndividualParameters
from cse599o_systems.ddp_overlap_bucketed import DDPOverlapBucketed
import torch.cuda.nvtx as nvtx

# Any necessary helper functions can be defined here.
def collect_and_put_results(rank, world_size, num_iters, comm_times, iteration_times, device,result_queue):
    # comm_t and iter_t are per-rank tensors (e.g., torch.tensor([ms], device=f"cuda:{rank}"))
    if num_iters > 0:
            comm_t = torch.tensor(comm_times, device=device, dtype=torch.float32)
            iter_t = torch.tensor(iteration_times, device=device, dtype=torch.float32)
    else:
        comm_t = torch.zeros(1, device=device, dtype=torch.float32)
        iter_t = torch.zeros(1, device=device, dtype=torch.float32)
    if rank == 0:
        comm_gather = [torch.empty_like(comm_t) for _ in range(world_size)]
        iter_gather = [torch.empty_like(iter_t) for _ in range(world_size)]
        dist.gather(comm_t, gather_list=comm_gather, dst=0)  # root provides gather_list
        dist.gather(iter_t, gather_list=iter_gather, dst=0)
    else:
        dist.gather(comm_t, dst=0)  # non-root: no gather_list
        dist.gather(iter_t, dst=0)

    if rank == 0:
        comm_results = [t.cpu().tolist() for t in comm_gather]
        iter_results = [t.cpu().tolist() for t in iter_gather]
        print(f"rank {rank} gathered communication times: {comm_results}")
        print(f"rank {rank} gathered iteration times: {iter_results}")
        result_queue.put((iter_results, comm_results))

# You can change the function and variable names as needed.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# ============================================================
# (0) Naive DDP
# ============================================================
# You can change the function and variable names as needed.
def run_naive(rank, world_size, config, num_iters, num_warmup, result_queue):
    """A naive DDP training loop for reference."""
    setup(rank, world_size)
    try:
        reset_seed(42)
        device = f"cuda:{rank}"
        print(f"Running naive DDP on rank {rank}.")
        # Construct model and move to GPU
        model = TransformerLM(
                vocab_size=config['model']['vocab_size'],
                context_length=config['model']['max_seq_len'],
                num_layers=config['model']['num_layers'],
                d_model=config['model']['d_model'],
                num_heads=config['model']['num_heads'],
                d_ff=config['model']['d_ff'],
                device=device,
                dtype=getattr(torch, config['model']['dtype'].split('.')[-1])
        ).to(device)
        optimizer = AdamW(model.parameters(), lr = config['optimizer']['max_lr'])
        model.train()

        rank_batch_size = config['training']['batch_size'] // world_size
        seq_len = config['model']['max_seq_len']
        comm_times = []
        iteration_times = []
        for step in range(num_iters + num_warmup):
            # create CUDA events for timing
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            comm_start_event = torch.cuda.Event(enable_timing=True)
            comm_end_event = torch.cuda.Event(enable_timing=True)
            # Dummy data
            input_ids = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)
            targets = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)

            # zero gradients
            optimizer.zero_grad()

            # ------ step timing starts ------
            step_start_event.record()

            # Forward pass
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, config['model']['vocab_size']), targets.view(-1))

            # Backward pass
            loss.backward()

            # ------ communication timing starts ------
            comm_start_event.record()

            # All-reduce gradients
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

            # ------ communication timing ends ------
            comm_end_event.record()

            optimizer.step()

            # ------ step timing ends ------
            step_end_event.record()
            torch.cuda.synchronize()

            if step >= num_warmup:
                iteration_time = step_start_event.elapsed_time(step_end_event)
                comm_time = comm_start_event.elapsed_time(comm_end_event)
                print(f"Rank {rank}, Step {step - num_warmup}: Iteration Time = {iteration_time} ms, Communication Time = {comm_time} ms")
                iteration_times.append(iteration_time)
                comm_times.append(comm_time)
            
        collect_and_put_results(rank, world_size, num_iters, comm_times, iteration_times, device,result_queue)
    finally:
        cleanup()


# ============================================================
# (1) Flat DDP
# ============================================================
# You can change the function and variable names as needed.
def run_flat(rank, world_size, config, num_iters, num_warmup, result_queue):
    """All-reduce a single flattened gradient tensor."""
    setup(rank, world_size)
    try:
        reset_seed(42)
        device = f"cuda:{rank}"
        print(f"Running flat DDP on rank {rank}.")
        # Construct model and move to GPU
        model = TransformerLM(
                vocab_size=config['model']['vocab_size'],
                context_length=config['model']['max_seq_len'],
                num_layers=config['model']['num_layers'],
                d_model=config['model']['d_model'],
                num_heads=config['model']['num_heads'],
                d_ff=config['model']['d_ff'],
                device=device,
                dtype=getattr(torch, config['model']['dtype'].split('.')[-1])
        ).to(device)
        optimizer = AdamW(model.parameters(), lr = config['optimizer']['max_lr'])
        model.train()

        rank_batch_size = config['training']['batch_size'] // world_size
        seq_len = config['model']['max_seq_len']
        comm_times = []
        iteration_times = []
        for step in range(num_iters + num_warmup):
            # create CUDA events for timing
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            comm_start_event = torch.cuda.Event(enable_timing=True)
            comm_end_event = torch.cuda.Event(enable_timing=True)
            # Dummy data
            input_ids = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)
            targets = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)

            # zero gradients
            optimizer.zero_grad()

            # ------ step timing starts ------
            step_start_event.record()

            # Forward pass
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, config['model']['vocab_size']), targets.view(-1))

            # Backward pass
            loss.backward()

            # ------ communication timing starts ------
            with nvtx.range("all-reduce comm"):
                comm_start_event.record()

                # All-reduce gradients
                # Flatten all gradients into a single tensor
                grads = [param.grad.data.view(-1) for param in model.parameters() if param.grad is not None]
                flat_grads = torch.cat(grads)
                # All-reduce the flattened gradient tensor
                dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG)
                # Unflatten the gradients back to their original shapes
                offset = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_shape = param.grad.data.shape
                        grad_numel = param.grad.data.numel()
                        param.grad.data.copy_(flat_grads[offset:offset + grad_numel].view(grad_shape))
                        offset += grad_numel

                # ------ communication timing ends ------
                comm_end_event.record()

            optimizer.step()

            # ------ step timing ends ------
            step_end_event.record()
            torch.cuda.synchronize()

            if step >= num_warmup:
                iteration_time = step_start_event.elapsed_time(step_end_event)
                comm_time = comm_start_event.elapsed_time(comm_end_event)
                print(f"Rank {rank}, Step {step - num_warmup}: Iteration Time = {iteration_time} ms, Communication Time = {comm_time} ms")
                iteration_times.append(iteration_time)
                comm_times.append(comm_time)
            
        collect_and_put_results(rank, world_size, num_iters, comm_times, iteration_times, device,result_queue)
    finally:
        cleanup()


# ============================================================
# (2) Individual DDP
# ============================================================
# You can change the function and variable names as needed.
def run_individual(rank, world_size, config, num_iters, num_warmup, result_queue):
    """All-reduce each parameter's gradient individually."""
    setup(rank, world_size)
    try:
        reset_seed(42)
        device = f"cuda:{rank}"
        print(f"Running individual DDP on rank {rank}.")
        # Construct model and move to GPU
        model = TransformerLM(
                vocab_size=config['model']['vocab_size'],
                context_length=config['model']['max_seq_len'],
                num_layers=config['model']['num_layers'],
                d_model=config['model']['d_model'],
                num_heads=config['model']['num_heads'],
                d_ff=config['model']['d_ff'],
                device=device,
                dtype=getattr(torch, config['model']['dtype'].split('.')[-1])
        ).to(device)
        ddp_model = DDPOverlapIndividualParameters(model)
        optimizer = AdamW(ddp_model.parameters(), lr = config['optimizer']['max_lr'])
        ddp_model.train()

        rank_batch_size = config['training']['batch_size'] // world_size
        seq_len = config['model']['max_seq_len']
        comm_times = []
        iteration_times = []
        for step in range(num_iters + num_warmup):
            # create CUDA events for timing
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            comm_start_event = torch.cuda.Event(enable_timing=True)
            comm_end_event = torch.cuda.Event(enable_timing=True)
            # Dummy data
            input_ids = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)
            targets = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)

            # zero gradients
            optimizer.zero_grad()

            # ------ step timing starts ------
            step_start_event.record()

            # Forward pass
            outputs = ddp_model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, config['model']['vocab_size']), targets.view(-1))

            # Backward pass
            loss.backward()

            # ------ communication timing starts ------
            comm_start_event.record()

            ddp_model.finish_gradient_synchronization()
            
            # ------ communication timing ends ------
            comm_end_event.record()

            optimizer.step()

            # ------ step timing ends ------
            step_end_event.record()
            torch.cuda.synchronize()

            if step >= num_warmup:
                iteration_time = step_start_event.elapsed_time(step_end_event)
                comm_time = comm_start_event.elapsed_time(comm_end_event)
                print(f"Rank {rank}, Step {step - num_warmup}: Iteration Time = {iteration_time} ms, Communication Time = {comm_time} ms")
                iteration_times.append(iteration_time)
                comm_times.append(comm_time)
            
        collect_and_put_results(rank, world_size, num_iters, comm_times, iteration_times, device,result_queue)
    finally:
        cleanup()

# ============================================================
# (3) Bucketed DDP
# ============================================================
# You can change the function and variable names as needed.
def run_bucketed(rank, world_size, config, num_iters, num_warmup, result_queue, bucket_size_mb):
    """Group gradients into buckets and all-reduce each bucket."""
    """All-reduce each parameter's gradient individually."""
    setup(rank, world_size)
    try:
        reset_seed(42)
        device = f"cuda:{rank}"
        print(f"Running bucketed DDP on rank {rank}.")
        # Construct model and move to GPU
        model = TransformerLM(
                vocab_size=config['model']['vocab_size'],
                context_length=config['model']['max_seq_len'],
                num_layers=config['model']['num_layers'],
                d_model=config['model']['d_model'],
                num_heads=config['model']['num_heads'],
                d_ff=config['model']['d_ff'],
                device=device,
                dtype=getattr(torch, config['model']['dtype'].split('.')[-1])
        ).to(device)
        ddp_model = DDPOverlapBucketed(model, bucket_size_mb=bucket_size_mb)
        optimizer = AdamW(ddp_model.parameters(), lr = config['optimizer']['max_lr'])
        ddp_model.train()

        rank_batch_size = config['training']['batch_size'] // world_size
        seq_len = config['model']['max_seq_len']
        comm_times = []
        iteration_times = []
        for step in range(num_iters + num_warmup):
            # create CUDA events for timing
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            comm_start_event = torch.cuda.Event(enable_timing=True)
            comm_end_event = torch.cuda.Event(enable_timing=True)
            # Dummy data
            input_ids = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)
            targets = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)

            # zero gradients
            optimizer.zero_grad()

            # ------ step timing starts ------
            step_start_event.record()

            # Forward pass
            outputs = ddp_model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, config['model']['vocab_size']), targets.view(-1))

            # Backward pass
            loss.backward()

            # ------ communication timing starts ------
            comm_start_event.record()

            ddp_model.finish_gradient_synchronization()
            
            # ------ communication timing ends ------
            comm_end_event.record()

            optimizer.step()

            # ------ step timing ends ------
            step_end_event.record()
            torch.cuda.synchronize()

            if step >= num_warmup:
                iteration_time = step_start_event.elapsed_time(step_end_event)
                comm_time = comm_start_event.elapsed_time(comm_end_event)
                print(f"Rank {rank}, Step {step - num_warmup}: Iteration Time = {iteration_time} ms, Communication Time = {comm_time} ms")
                iteration_times.append(iteration_time)
                comm_times.append(comm_time)
            
        collect_and_put_results(rank, world_size, num_iters, comm_times, iteration_times, device,result_queue)
    finally:
        cleanup()


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
        choices=["naive", "flat", "individual", "bucketed"],
        help="Select which DDP variant to benchmark.",
    )
    parser.add_argument(
        "--bucket-size-mb",
        type=int,
        default=10,
        help="Bucket size (in MB) for the bucketed DDP variant.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../cse599o_basics/config.json",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    # Example placeholders
    # num_iters is the number of benchmark iterations
    # num_warmup is the number of warmup iterations
    num_iters, num_warmup = 5, 5
    
    # DDP setup
    mp.set_start_method("spawn", force=True)
    world_size = 2
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    # Load configuration
    config = load_config(args.config)

    if args.mode == "naive":
        mp.spawn(run_naive, args=(world_size, config, num_iters, num_warmup, result_queue), nprocs=world_size, join=True)
    elif args.mode == "flat":
        mp.spawn(run_flat, args=(world_size, config, num_iters, num_warmup, result_queue), nprocs=world_size, join=True)
    elif args.mode == "individual":
        mp.spawn(run_individual, args=(world_size, config, num_iters, num_warmup, result_queue), nprocs=world_size, join=True)
    elif args.mode == "bucketed":
        mp.spawn(run_bucketed, args=(world_size, config, num_iters, num_warmup, result_queue, args.bucket_size_mb), nprocs=world_size, join=True)

    print(f"Mode: {args.mode}")
    iteration_times, comm_times = result_queue.get()
    print(f"Iteration Times (ms): {iteration_times}")
    print(f"Communication Times (ms): {comm_times}")
    # take the max across all ranks for each iteration
    max_iteration_times = [max(times) for times in zip(*iteration_times)]
    max_comm_times = [max(times) for times in zip(*comm_times)]
    avg_iteration_time = sum(max_iteration_times) / len(max_iteration_times)
    avg_comm_time = sum(max_comm_times) / len(max_comm_times)
    print(f"Average Iteration Time (ms): {avg_iteration_time}")
    print(f"Average Communication Time (ms): {avg_comm_time}")


if __name__ == "__main__":
    benchmark_optimized_ddp()
