# sharding_optimizer.py
# -------------------------------------------------------------
# CSE 599O: 
#
# Implement optimizer state sharding for distributed training.
#
# -------------------------------------------------------------
import os
import torch
import torch.distributed as dist
import argparse
import json
import torch.multiprocessing as mp
from multiprocessing import Manager
from timeit import default_timer as timer
# You can add other necessary imports here.
from optimizer_state_sharding import ShardedStateOptimizer
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.optimizer import AdamW


# Add any necessary helper functions here.
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


# You can change the function and variable names as needed.
def run_distributed_training(rank, world_size, config, num_trials, num_warmup_trials, result_queue, use_sharding):
    # Setup distributed environment
    setup(rank, world_size)
    try:
        device = f"cuda:{rank}"
        print(f"Running distributed training on rank {rank}.")

        # Construct model
        if rank == 0:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
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
        if rank == 0:
            print(torch.cuda.memory_allocated() / (1024**2), "MB after model construction")
        if use_sharding:
            optimizer = ShardedStateOptimizer(
                model.parameters(),
                optimizer_cls=AdamW,
                lr=config['optimizer']['max_lr']
            )
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=config['optimizer']['max_lr']
            )
        model.train()

        rank_batch_size = config['training']['batch_size'] // world_size
        seq_len = config['model']['max_seq_len']

        iteration_times = []

        for step in range(num_warmup_trials + num_trials):
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            # Construct random input data
            input_ids = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)
            targets = torch.randint(0, config['model']['vocab_size'], (rank_batch_size, seq_len), device=device)

            if step >= num_warmup_trials:
                step_start_event.record()

            optimizer.zero_grad()

            outputs = model(input_ids)
            if rank == 0:
                print(torch.cuda.memory_allocated() / (1024**2), "MB before loss computation")
                print(torch.cuda.memory_reserved() / (1024**2), "MB reserved before loss computation")
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, config['model']['vocab_size']),
                targets.view(-1)
            )

            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            
            if rank == 0:
                print(torch.cuda.memory_allocated() / (1024**2), "MB before optimizer step")
                print(torch.cuda.memory_reserved() / (1024**2), "MB reserved before optimizer step")
            optimizer.step()
            if rank == 0:
                print(torch.cuda.memory_allocated() / (1024**2), "MB after optimizer step")
                print(torch.cuda.memory_reserved() / (1024**2), "MB reserved after optimizer step")
            
            if step >= num_warmup_trials:
                step_end_event.record()
                torch.cuda.synchronize()
                iteration_time = step_start_event.elapsed_time(step_end_event)  # in milliseconds
                iteration_times.append(iteration_time)

            print(f"Rank {rank}, Step {step}, Loss: {loss.item()}")
        if rank == 0:
            torch.cuda.memory._dump_snapshot("rank0_memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
        
        if num_trials > 0:
            iter_t = torch.tensor(iteration_times, device=device, dtype=torch.float32)
            if rank == 0:
                iter_gather = [torch.zeros_like(iter_t) for _ in range(world_size)]
                dist.gather(iter_t, gather_list=iter_gather, dst=0)
            else:
                dist.gather(iter_t, dst=0)
        
        if rank == 0:
            iter_results = [t.cpu().tolist() for t in iter_gather]
            print(f"rank {rank} gathered iteration times: {iter_results}")
            result_queue.put(iter_results)
    finally:
        cleanup()
        
if __name__ == "__main__":
    # Set up distributed training parameters
    # Collect results and print timing summary
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../cse599o_basics/config.json",
        help="Path to the configuration file.",
    )
    parser.add_argument("--optimizer_sharding", action="store_true", help="Use optimizer state sharding.")
    args = parser.parse_args()
    
    world_size = 2
    num_timing_iters, num_warmup_iters = 5, 5

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    config = load_config(args.config)

    mp.spawn(run_distributed_training,
             args=(world_size, config, num_timing_iters, num_warmup_iters, result_queue, args.optimizer_sharding),
             nprocs=world_size,
             join=True)

    iteration_times = result_queue.get()  # Get rank 0 completion message
    print(f"Result message: {iteration_times}")
    max_iteration_times = [max(times) for times in zip(*iteration_times)]
    print(f"Per-iteration times (ms): {max_iteration_times}")
    avg_iteration_times = sum(max_iteration_times) / len(max_iteration_times)
    print(f"optimizer sharding {args.optimizer_sharding} Average iteration time over {num_timing_iters} iterations: {avg_iteration_times:.2f} ms")