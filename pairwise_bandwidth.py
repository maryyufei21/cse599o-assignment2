#!/usr/bin/env python3
"""
Pairwise GPU<->GPU bandwidth benchmark (PyTorch Distributed + NCCL).

For each GPU pair (i, j), this script:
  • tests multiple message sizes (MiB),
  • measures one-way bandwidth in both directions (i→j and j→i),
  • records the maximum GB/s seen across all sizes & directions,
  • prints an NxN table of GB/s (diagonal is '-').

Usage:
  torchrun --nproc_per_node=8 pairwise_bandwidth.py \
    --sizes-mib 8 64 256 512 \
    --iters 30 \
    --warmup-iters 5

Notes:
  • Use NCCL backend on a single node with CUDA-visible GPUs.
  • Results depend on topology (NVLink/PCIe), driver & NCCL versions.
"""

import argparse
import os
import time
from typing import List

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="Pairwise GPU bandwidth benchmark")
    p.add_argument(
        "--backend", type=str, default="nccl",
        help="Process group backend (default: nccl)"
    )
    p.add_argument(
        "--sizes-mib", type=int, nargs="+", default=[8, 64, 256],
        help="Message sizes to test (in MiB), e.g. 8 64 256"
    )
    p.add_argument(
        "--iters", type=int, default=30,
        help="Measurement iterations per size/direction (default: 30)"
    )
    p.add_argument(
        "--warmup-iters", type=int, default=5,
        help="Warmup iterations per size/direction (default: 5)"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-size and per-direction details"
    )
    return p.parse_args()


def init_distributed(backend: str):
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Map this process to its local GPU. torchrun sets LOCAL_RANK.
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, device, local_rank


@torch.inference_mode()
def measure_unidirectional_bw(
    send_rank: int,
    recv_rank: int,
    nbytes: int,
    iters: int,
    warmup_iters: int,
    device: torch.device,
    verbose: bool = False,
) -> float:
    """
    Measure one-way bandwidth (GB/s) from send_rank -> recv_rank
    using torch.distributed isend/irecv with CUDA tensors.

    Returns the measured GB/s (same value on all ranks via broadcast).
    """
    rank = dist.get_rank()

    # Use uint8 so "elements == bytes" (avoids dtype rounding).
    # Allocate buffers only on participating ranks.
    send_buf = None
    recv_buf = None
    if rank == send_rank:
        send_buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
    if rank == recv_rank:
        recv_buf = torch.empty(nbytes, dtype=torch.uint8, device=device)

    # Everyone enters the same number of barriers.
    dist.barrier()

    # Warmup
    for _ in range(warmup_iters):
        if rank == send_rank:
            work = dist.isend(tensor=send_buf, dst=recv_rank)
            work.wait()
        elif rank == recv_rank:
            work = dist.irecv(tensor=recv_buf, src=send_rank)
            work.wait()
        dist.barrier()

    torch.cuda.synchronize(device)
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        if rank == send_rank:
            work = dist.isend(tensor=send_buf, dst=recv_rank)
            work.wait()
        elif rank == recv_rank:
            work = dist.irecv(tensor=recv_buf, src=send_rank)
            work.wait()
        dist.barrier()
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-9)
    # Decimal GB/s (1e9 bytes) for readability; change to GiB if preferred.
    gbps = (nbytes * iters) / elapsed / 1e9

    # Broadcast the measured GB/s from the receiver so every rank has it.
    gbps_tensor = torch.tensor(
        [gbps if rank == recv_rank else 0.0],
        dtype=torch.float32,
        device=device,
    )
    dist.broadcast(gbps_tensor, src=recv_rank)
    gbps_out = float(gbps_tensor.item())

    if verbose and rank == 0:
        print(f"  size={nbytes/2**20:.0f} MiB, {send_rank}->{recv_rank}: {gbps_out:.3f} GB/s")

    return gbps_out


def print_csv_table(matrix: List[List[float]]):
    """Print NxN CSV table of GB/s with '-' on the diagonal."""
    n = len(matrix)
    header = ["gpu"] + [f"gpu{j}" for j in range(n)]
    print(",".join(header))
    for i in range(n):
        row = [f"gpu{i}"]
        for j in range(n):
            if i == j:
                row.append("-")
            else:
                row.append(f"{matrix[i][j]:.3f}")
        print(",".join(row))


def main():
    args = parse_args()
    rank, world_size, device, local_rank = init_distributed(args.backend)

    if world_size < 2:
        if rank == 0:
            print("Need at least 2 processes.")
        return

    if rank == 0:
        print(f"World size: {world_size} | Backend: {args.backend}")
        print(f"Testing sizes (MiB): {args.sizes_mib}")
        print(f"Iters={args.iters}, Warmup={args.warmup_iters}\n")

    # Convert sizes to bytes (MiB -> bytes).
    sizes_bytes = [mib * (2 ** 20) for mib in args.sizes_mib]

    # Prepare result matrix (NxN), filled with 0.0; diagonal will be ignored.
    result = [[0.0 for _ in range(world_size)] for _ in range(world_size)]

    # Synchronize before measurements start.
    dist.barrier()

    # Loop over unique pairs (i < j). For each pair, test multiple sizes
    # and both directions, then record the maximum GB/s observed.
    for i in range(world_size):
        for j in range(i + 1, world_size):
            max_gbps = 0.0
            if rank == 0 and args.verbose:
                print(f"Pair (gpu{i}, gpu{j})")

            for nbytes in sizes_bytes:
                bw_ij = measure_unidirectional_bw(
                    i, j, nbytes, args.iters, args.warmup_iters, device, verbose=args.verbose
                )
                bw_ji = measure_unidirectional_bw(
                    j, i, nbytes, args.iters, args.warmup_iters, device, verbose=args.verbose
                )
                max_gbps = max(max_gbps, bw_ij, bw_ji)

            # Ensure all ranks agree on the same max (they should).
            # Broadcast max_gbps from rank 0 for consistency.
            max_tensor = torch.tensor([max_gbps], dtype=torch.float32, device=device)
            dist.broadcast(max_tensor, src=0)
            max_gbps = float(max_tensor.item())

            # Only rank 0 fills in and prints.
            if rank == 0:
                result[i][j] = max_gbps
                result[j][i] = max_gbps

    dist.barrier()

    if rank == 0:
        print("Pairwise max bandwidth table (GB/s):")
        print_csv_table(result)
        print("\nDone.")


if __name__ == "__main__":
    main()
