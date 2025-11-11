import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def load_results(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            # Strip whitespace from keys and values to handle spaces after commas
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in raw_row.items()}

            # Skip empty lines
            if not row.get("world_size"):
                continue

            row["backend"] = row["backend"]
            row["world_size"] = int(row["world_size"])
            row["message_bytes"] = int(row["message_bytes"])
            row["message_mb"] = float(row["message_mb"])
            row["iter_max_mean_ms"] = float(row["iter_max_mean_ms"])
            row["avg_of_rank_means_ms"] = float(row["avg_of_rank_means_ms"])

            # ref_link_gbps column may be empty
            ref = row.get("ref_link_gbps", "")
            row["ref_link_gbps_val"] = float(ref) if ref not in (None, "",) else None

            rows.append(row)
    return rows


def compute_throughput(row, use_max_time: bool) -> float:
    """
    Effective (bus) bandwidth in Gb/s for a ring all-reduce:

        bytes_moved_per_rank = 2 * (p-1)/p * N

    Throughput = bytes_moved_per_rank / time / 1e9
    """
    p = row["world_size"]
    N = row["message_bytes"]

    comm_bytes = 2.0 * (p - 1) / p * N

    if use_max_time:
        t_ms = row["iter_max_mean_ms"]
    else:
        t_ms = row["avg_of_rank_means_ms"]

    t_s = t_ms / 1e3
    return (comm_bytes / t_s) / 1e9  # GB/s


def group_by_world_size(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["world_size"]].append(r)
    # sort each group by message size (MB)
    for ws in groups:
        groups[ws].sort(key=lambda r: r["message_mb"])
    return groups


def plot_throughput(groups, use_max_time, ref_link_gbps, out_path):
    """
    Plot:
      X: message size (MB, log scale)
      Y: throughput (GB/s)
      One curve per world_size
      Horizontal theoretical reference line at ref_link_gbps
    """
    plt.figure()

    label_suffix = " (max time across ranks)" if use_max_time else " (avg time across ranks)"

    for world_size, rows in sorted(groups.items()):
        xs = [r["message_mb"] for r in rows]
        ys = [compute_throughput(r, use_max_time=use_max_time) for r in rows]
        plt.plot(xs, ys, marker="o", label=f"{world_size} GPUs")

    plt.xscale("log")
    # Nice ticks for 1, 10, 100, 1024 MB
    plt.xticks([1, 10, 100, 1024], ["1", "10", "100", "1024"])
    plt.xlabel("Message size (MB)")
    plt.ylabel("Throughput (GB/s)")
    plt.title("All-reduce throughput" + label_suffix)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()

    if ref_link_gbps is not None:
        # Draw horizontal theoretical line over x-range
        all_xs = []
        for rows in groups.values():
            all_xs.extend([r["message_mb"] for r in rows])
        if all_xs:
            xmin, xmax = min(all_xs), max(all_xs)
            plt.hlines(ref_link_gbps, xmin, xmax, linestyles="dashed")
            plt.text(
                xmin,
                ref_link_gbps,
                f"  theoretical {ref_link_gbps} Gb/s",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="problem-1-allreduce_results2.csv", help="CSV file with benchmark results")
    ap.add_argument(
        "--ref-link-gbps",
        type=float,
        # required=True,
        default=None,
        help=(
            "Theoretical per-GPU link bandwidth in Gb/s, used for the reference curve "
            "(e.g., from NVLink/NVSwitch/PCIe specs)."
        ),
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="problem-1-allreduce_throughput",
        help="Prefix for output image files",
    )
    args = ap.parse_args()

    rows = load_results(args.csv_path)
    groups = group_by_world_size(rows)

    # Main figure: throughput based on MAX time across ranks
    plot_throughput(
        groups,
        use_max_time=True,
        ref_link_gbps=args.ref_link_gbps,
        out_path=f"{args.out_prefix}_max.png",
    )

    # Secondary figure: throughput based on AVERAGE time across ranks
    plot_throughput(
        groups,
        use_max_time=False,
        ref_link_gbps=args.ref_link_gbps,
        out_path=f"{args.out_prefix}_avg.png",
    )


if __name__ == "__main__":
    main()
