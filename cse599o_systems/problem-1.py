import os, time, argparse, csv, statistics
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def parse_size(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("GB"): return int(float(s[:-2]) * (1024**3))
    if s.endswith("MB"): return int(float(s[:-2]) * (1024**2))
    if s.endswith("KB"): return int(float(s[:-2]) * (1024**1))
    if s.endswith("B"):  return int(float(s[:-1]))
    # bare number => MB
    return int(float(s) * (1024**2))

def setup(rank: int, world_size: int, backend: str):
    # Single-node defaults; for multi-node, set MASTER_ADDR/PORT and ranks explicitly.
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

@torch.inference_mode()
def time_allreduce(tensor, backend: str) -> float:
    """Return elapsed milliseconds of a single all_reduce, correctly synchronized."""
    start = torch.cuda.Event(enable_timing=True) if backend == "nccl" else None
    end = torch.cuda.Event(enable_timing=True) if backend == "nccl" else None
    if backend == "nccl":
        start.record()
    else:
        start = time.perf_counter()
    dist.all_reduce(tensor)  # in-place, SUM by default
    
    # Ensure the op actually finished before stopping the clock.
    if backend == "nccl":
        end.record()
        torch.cuda.synchronize(tensor.device)
        elapsed_ms = start.elapsed_time(end)
        return elapsed_ms
    else:
        end = time.perf_counter()
        return (end - start) * 1e3  # ms

def run_rank(rank: int, world_size: int, sizes_b, iters: int, warmup: int,
             backend: str, outfile: str | None, link_gbps: float | None):
    setup(rank, world_size, backend)

    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    dtype = torch.float32

    # Each (size, world_size) benchmark
    rows_to_write = []
    for N in sizes_b:
        numel = N // 4  # float32 bytes-per-elt
        if numel <= 0:
            if rank == 0: print(f"Skipping size {N} (too small).")
            continue

        # Allocate outside timed region
        x = torch.ones(numel, device=device, dtype=dtype)

        # Warm-up (>=5 as recommended)
        for _ in range(warmup):
            _ = time_allreduce(x, backend)

        # Timed iterations
        per_iter_max_ms = []          # max across ranks each iter (the “true” iter time)
        per_rank_local_ms = []        # this rank's raw times (for spread/statistics)

        for _ in range(iters):
            t_ms_local = time_allreduce(x, backend)
            per_rank_local_ms.append(t_ms_local)

            # Compute per-iteration MAX across ranks efficiently
            # Use a 1-element tensor on the correct device for the all_reduce
            t_tensor = torch.tensor([t_ms_local], device=device, dtype=torch.float32)
            dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX)
            per_iter_max_ms.append(float(t_tensor.item()))

        # Core stats based on per-iteration MAX across ranks
        mean_ms = float(sum(per_iter_max_ms) / len(per_iter_max_ms))
        median_ms = float(statistics.median(per_iter_max_ms))

        # Effective (bus) bandwidth per rank for ring all-reduce:
        # bytes moved per rank = 2*(p-1)/p * N
        comm_bytes = 2.0 * (world_size - 1) / world_size * N
        thrpt_gbps = (comm_bytes / (mean_ms / 1e3)) / 1e9

        # Optionally collect per-rank summaries to examine variability
        # (not needed to compute throughput, but useful per guideline)
        local_summary = {
            "rank": rank,
            "mean_ms_local": float(sum(per_rank_local_ms) / len(per_rank_local_ms)),
            "median_ms_local": float(statistics.median(per_rank_local_ms)),
        }
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_summary)
        # Reduce to cross-rank aggregates on rank 0
        if rank == 0:
            avg_of_rank_means = float(sum(g["mean_ms_local"] for g in gathered) / world_size)
            med_of_rank_meds = float(statistics.median([g["median_ms_local"] for g in gathered]))
            ref_gbps = float(link_gbps) if link_gbps is not None else None

            rows_to_write.append({
                "backend": backend,
                "world_size": world_size,
                "message_bytes": N,
                "message_mb": round(N / (1024**2), 3),
                # “truth” for a collective is the per-iter max across ranks:
                "iter_max_mean_ms": round(mean_ms, 3),
                "iter_max_median_ms": round(median_ms, 3),
                "throughput_gbps": round(thrpt_gbps, 3),
                # diagnostic aggregates across ranks:
                "avg_of_rank_means_ms": round(avg_of_rank_means, 3),
                "median_of_rank_medians_ms": round(med_of_rank_meds, 3),
                # reference flat line (optional)
                "ref_link_gbps": ref_gbps,
            })

    if rank == 0:
        if not rows_to_write:
            print("No results to write.")
        else:
            if outfile:
                write_header = not os.path.exists(outfile)
                with open(outfile, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows_to_write[0].keys()))
                    if write_header:
                        w.writeheader()
                    w.writerows(rows_to_write)
                print(f"Wrote {len(rows_to_write)} rows to {outfile}")
            else:
                for r in rows_to_write:
                    print(r)

    cleanup()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, required=True, choices=[2,4,8])
    ap.add_argument("--backend", type=str, default="nccl", choices=["nccl","gloo"],
                    help="Use gloo+CPU for local debugging; nccl+GPU for benchmarking.")
    ap.add_argument("--sizes", nargs="+", default=["1MB","10MB","100MB","1GB"])
    ap.add_argument("--iters", type=int, default=30, help="Timed iterations per size (keep <5 minutes).")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations per size.")
    ap.add_argument("--outfile", type=str, default="problem-1-allreduce_results2.csv")
    ap.add_argument("--link-gbps", type=float, default=None,
                    help="Per-GPU link bandwidth for reference curve (e.g., 300 for NVSwitch).")
    args = ap.parse_args()

    sizes_b = [parse_size(s) for s in args.sizes]

    # Important: keep all runs on the SAME machine for fair comparisons.
    mp.spawn(
        run_rank,
        args=(args.world_size, sizes_b, args.iters, args.warmup, args.backend, args.outfile, args.link_gbps),
        nprocs=args.world_size,
        join=True,
    )

if __name__ == "__main__":
    main()
