#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to generate profiler plots. "
        "Install it with `pip install matplotlib`."
    ) from exc

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


DISTTGL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = DISTTGL_ROOT / "profiles" / "plots"
DEFAULT_BATCH_SIZES = [1024, 2048, 4096, 8192, 16384, 32768]

SYSTEM_COLORS = {
    "GNNFlow": "#1f77b4",
    "DistTGL": "#ff7f0e",
}

SYSTEM_MARKERS = {
    "GNNFlow": "o",
    "DistTGL": "^",
}

TRAIN_STAGE_ORDER = [
    ("sampling", "Sampling"),
    ("feature_fetch", "Feature Fetch"),
    ("memory_fetch", "Memory Fetch"),
    ("memory_update", "Memory Update"),
    ("memory_write_back", "Memory Write Back"),
    ("model_forward", "Model Forward"),
    ("loss_backward_optimizer", "Loss+Backward+Opt"),
]

TRAIN_STAGE_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
]


@dataclass
class ProfileRun:
    system: str
    summary_path: Path
    modified_at: float
    model: str
    dataset: str
    cache: str
    edge_cache_ratio: float
    node_cache_ratio: float
    snapshot_time_window: float
    batch_size: int
    world_size: int
    step_time_avg_sec: Optional[float]
    throughput_avg: Optional[float]
    peak_allocated_bytes: Optional[float]
    peak_reserved_bytes: Optional[float]
    gpu_load_avg_pct: Optional[float]
    gpu_memory_util_avg_pct: Optional[float]
    gpu_memory_used_avg_mb: Optional[float]
    cpu_max_rss_bytes: Optional[float]
    train_stage_time_sec: Dict[str, Optional[float]]


def nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_avg(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        return as_float(value.get("avg"))
    return as_float(value)


def metric_max(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        return as_float(value.get("max"))
    return as_float(value)


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) == 0:
        return None
    return sum(numeric_values) / len(numeric_values)


def maximum(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) == 0:
        return None
    return max(numeric_values)


def maybe_gib(value_bytes: Optional[float]) -> Optional[float]:
    if value_bytes is None:
        return None
    return float(value_bytes) / (1024 ** 3)


def discover_summary_paths(profiles_dir: Path) -> List[Path]:
    if not profiles_dir.exists():
        return []

    summary_paths: List[Path] = []
    for run_dir in sorted(path for path in profiles_dir.iterdir() if path.is_dir()):
        aggregate_summary = run_dir / "summary_all_ranks.json"
        rank0_summary = run_dir / "rank0" / "summary.json"
        if aggregate_summary.exists():
            summary_paths.append(aggregate_summary)
        elif rank0_summary.exists():
            summary_paths.append(rank0_summary)
    return summary_paths


def build_profile_run(summary_path: Path, system: str) -> ProfileRun:
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    rank_summaries = summary.get("ranks")
    if not isinstance(rank_summaries, list) or len(rank_summaries) == 0:
        rank_summaries = [summary]

    train_stage_time_sec = {
        key: metric_avg(nested_get(summary, "stage_time_sec", key))
        if nested_get(summary, "stage_time_sec", key) is not None
        else average(
            metric_avg(nested_get(rank_summary, "stage_time_sec", key))
            for rank_summary in rank_summaries
        )
        for key, _ in TRAIN_STAGE_ORDER
    }

    gpu_memory_used_avg_mb = metric_avg(summary.get("gpu_memory_used_mb"))
    if gpu_memory_used_avg_mb is None:
        gpu_memory_used_avg_mb = average(
            as_float(nested_get(rank_summary, "gpu_monitor", "avg_memory_used_mb"))
            for rank_summary in rank_summaries
        )

    cpu_max_rss_bytes = metric_max(summary.get("cpu_max_rss_bytes"))
    if cpu_max_rss_bytes is None:
        cpu_max_rss_bytes = maximum(
            as_float(nested_get(rank_summary, "cpu_memory", "max_rss_bytes"))
            for rank_summary in rank_summaries
        )

    peak_allocated_bytes = metric_max(summary.get("peak_allocated_bytes"))
    if peak_allocated_bytes is None:
        peak_allocated_bytes = maximum(
            as_float(nested_get(rank_summary, "cuda_memory", "peak_allocated_bytes"))
            for rank_summary in rank_summaries
        )

    peak_reserved_bytes = metric_max(summary.get("peak_reserved_bytes"))
    if peak_reserved_bytes is None:
        peak_reserved_bytes = maximum(
            as_float(nested_get(rank_summary, "cuda_memory", "peak_reserved_bytes"))
            for rank_summary in rank_summaries
        )

    gpu_load_avg_pct = metric_avg(summary.get("gpu_load_pct"))
    if gpu_load_avg_pct is None:
        gpu_load_avg_pct = average(
            as_float(nested_get(rank_summary, "gpu_monitor", "avg_load_pct"))
            for rank_summary in rank_summaries
        )

    gpu_memory_util_avg_pct = metric_avg(summary.get("gpu_memory_util_pct"))
    if gpu_memory_util_avg_pct is None:
        gpu_memory_util_avg_pct = average(
            as_float(nested_get(rank_summary, "gpu_monitor", "avg_memory_util_pct"))
            for rank_summary in rank_summaries
        )

    return ProfileRun(
        system=system,
        summary_path=summary_path,
        modified_at=summary_path.stat().st_mtime,
        model=str(summary["model"]),
        dataset=str(summary["dataset"]),
        cache=str(summary.get("cache", "unknown")),
        edge_cache_ratio=float(summary.get("edge_cache_ratio", 0.0)),
        node_cache_ratio=float(summary.get("node_cache_ratio", 0.0)),
        snapshot_time_window=float(summary.get("snapshot_time_window", 0.0)),
        batch_size=int(summary["batch_size"]),
        world_size=int(summary.get("world_size", 1)),
        step_time_avg_sec=metric_avg(summary.get("step_time_sec")),
        throughput_avg=metric_avg(summary.get("throughput_samples_per_sec")),
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        gpu_load_avg_pct=gpu_load_avg_pct,
        gpu_memory_util_avg_pct=gpu_memory_util_avg_pct,
        gpu_memory_used_avg_mb=gpu_memory_used_avg_mb,
        cpu_max_rss_bytes=cpu_max_rss_bytes,
        train_stage_time_sec=train_stage_time_sec,
    )


def collect_runs(profiles_dir: Path, system: str) -> List[ProfileRun]:
    return [
        build_profile_run(summary_path, system)
        for summary_path in discover_summary_paths(profiles_dir)
    ]


def latest_run_per_batch(
        runs: Sequence[ProfileRun],
        *,
        model: str,
        dataset: str,
        batch_sizes: Sequence[int],
        world_size: Optional[int],
        cache: Optional[str] = None,
        edge_cache_ratio: Optional[float] = None,
        node_cache_ratio: Optional[float] = None,
        snapshot_time_window: Optional[float] = None) -> Dict[int, ProfileRun]:
    selected: Dict[int, ProfileRun] = {}
    target_batch_sizes = set(batch_sizes)

    for run in runs:
        if run.model != model or run.dataset != dataset:
            continue
        if run.batch_size not in target_batch_sizes:
            continue
        if world_size is not None and run.world_size != world_size:
            continue
        if cache is not None and run.cache != cache:
            continue
        if edge_cache_ratio is not None and \
                not math.isclose(run.edge_cache_ratio, edge_cache_ratio, rel_tol=0, abs_tol=1e-9):
            continue
        if node_cache_ratio is not None and \
                not math.isclose(run.node_cache_ratio, node_cache_ratio, rel_tol=0, abs_tol=1e-9):
            continue
        if snapshot_time_window is not None and \
                not math.isclose(run.snapshot_time_window, snapshot_time_window, rel_tol=0, abs_tol=1e-9):
            continue

        current = selected.get(run.batch_size)
        if current is None or run.modified_at > current.modified_at:
            selected[run.batch_size] = run

    return selected


def write_selected_runs_csv(rows: Sequence[ProfileRun], output_path: Path):
    field_names = [
        "system",
        "model",
        "dataset",
        "cache",
        "batch_size",
        "world_size",
        "edge_cache_ratio",
        "node_cache_ratio",
        "snapshot_time_window",
        "step_time_avg_sec",
        "throughput_avg",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "gpu_load_avg_pct",
        "gpu_memory_util_avg_pct",
        "gpu_memory_used_avg_mb",
        "cpu_max_rss_bytes",
        "summary_path",
    ] + [f"train_{stage_key}_sec" for stage_key, _ in TRAIN_STAGE_ORDER]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for run in rows:
            row = {
                "system": run.system,
                "model": run.model,
                "dataset": run.dataset,
                "cache": run.cache,
                "batch_size": run.batch_size,
                "world_size": run.world_size,
                "edge_cache_ratio": run.edge_cache_ratio,
                "node_cache_ratio": run.node_cache_ratio,
                "snapshot_time_window": run.snapshot_time_window,
                "step_time_avg_sec": run.step_time_avg_sec,
                "throughput_avg": run.throughput_avg,
                "peak_allocated_bytes": run.peak_allocated_bytes,
                "peak_reserved_bytes": run.peak_reserved_bytes,
                "gpu_load_avg_pct": run.gpu_load_avg_pct,
                "gpu_memory_util_avg_pct": run.gpu_memory_util_avg_pct,
                "gpu_memory_used_avg_mb": run.gpu_memory_used_avg_mb,
                "cpu_max_rss_bytes": run.cpu_max_rss_bytes,
                "summary_path": str(run.summary_path),
            }
            for stage_key, _ in TRAIN_STAGE_ORDER:
                row[f"train_{stage_key}_sec"] = run.train_stage_time_sec.get(stage_key)
            writer.writerow(row)


def format_metric_label(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_overview_comparison(
        batch_sizes: Sequence[int],
        system_runs: Dict[str, Dict[int, ProfileRun]],
        output_path: Path,
        title_context: str):
    metric_specs = [
        ("Step Time", "ms",
         lambda run: None if run.step_time_avg_sec is None
         else run.step_time_avg_sec * 1000.0),
        ("Throughput", "samples/s", lambda run: run.throughput_avg),
        ("Peak Allocated", "GiB", lambda run: maybe_gib(run.peak_allocated_bytes)),
        ("Peak Reserved", "GiB", lambda run: maybe_gib(run.peak_reserved_bytes)),
        ("Avg GPU Load", "%", lambda run: run.gpu_load_avg_pct),
        ("Avg GPU Mem Util", "%", lambda run: run.gpu_memory_util_avg_pct),
        ("Avg GPU Mem Used", "MB", lambda run: run.gpu_memory_used_avg_mb),
        ("Peak RSS", "GiB", lambda run: maybe_gib(run.cpu_max_rss_bytes)),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(19.5, 9.2), constrained_layout=False)
    axes_list = list(axes.flat)

    for axis, (title, ylabel, accessor) in zip(axes_list, metric_specs):
        for system_name in ("GNNFlow", "DistTGL"):
            runs = system_runs[system_name]
            x_values: List[int] = []
            y_values: List[float] = []
            for batch_size in batch_sizes:
                run = runs.get(batch_size)
                if run is None:
                    continue
                raw_value = accessor(run)
                if raw_value is None:
                    continue
                x_values.append(batch_size)
                y_values.append(float(raw_value))
            if len(y_values) == 0:
                continue
            axis.plot(
                x_values, y_values,
                color=SYSTEM_COLORS[system_name],
                marker=SYSTEM_MARKERS[system_name],
                linewidth=2.2,
                markersize=6,
                label=system_name,
            )
        axis.set_title(title)
        axis.set_xlabel("Batch Size")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(batch_sizes)
        axis.set_xticklabels([str(value) for value in batch_sizes])
        axis.grid(True, alpha=0.22)

    handles = [
        Patch(facecolor=SYSTEM_COLORS[system_name], label=system_name)
        for system_name in ("GNNFlow", "DistTGL")
    ]
    figure.legend(
        handles, [handle.get_label() for handle in handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=2, frameon=False, columnspacing=1.8)
    figure.text(0.5, 0.968, title_context, ha="center", va="top", fontsize=10)
    figure.suptitle("TGN System Comparison", fontsize=16, y=0.992)
    figure.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.99,
                           wspace=0.26, hspace=0.28)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_stage_breakdown_comparison(
        batch_sizes: Sequence[int],
        system_runs: Dict[str, Dict[int, ProfileRun]],
        output_path: Path,
        title_context: str):
    systems = ("GNNFlow", "DistTGL")
    figure, axes = plt.subplots(1, 2, figsize=(18, 6.8), sharey=True,
                                constrained_layout=False)

    for axis, system_name in zip(axes, systems):
        runs = system_runs[system_name]
        x_positions = list(range(len(batch_sizes)))
        bottoms = [0.0] * len(batch_sizes)

        for color, (stage_key, stage_label) in zip(TRAIN_STAGE_COLORS, TRAIN_STAGE_ORDER):
            values: List[float] = []
            for batch_size in batch_sizes:
                run = runs.get(batch_size)
                value = None if run is None else run.train_stage_time_sec.get(stage_key)
                values.append(0.0 if value is None else float(value) * 1000.0)
            axis.bar(
                x_positions, values, bottom=bottoms, color=color,
                width=0.72, label=stage_label)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

        axis.set_title(system_name)
        axis.set_xlabel("Batch Size")
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
        axis.grid(True, axis="y", alpha=0.22)

    axes[0].set_ylabel("Avg Stage Time per Profiled Step (ms)")
    figure.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=4, frameon=False, columnspacing=1.2)
    figure.text(0.5, 0.968, title_context, ha="center", va="top", fontsize=10)
    figure.suptitle("TGN Training Stage Breakdown", fontsize=16, y=0.992)
    figure.subplots_adjust(top=0.80, bottom=0.12, left=0.07, right=0.995,
                           wspace=0.08)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GNNFlow vs DistTGL profiler summaries for TGN.")
    parser.add_argument(
        "--gnnflow-root", type=Path, required=True,
        help="Path to the local GNNFlow repository root.")
    parser.add_argument(
        "--dataset", type=str, default="REDDIT",
        help="Dataset to compare (default: %(default)s)")
    parser.add_argument(
        "--model", type=str, default="TGN",
        help="Model label to compare (default: %(default)s)")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to compare (default: %(default)s)")
    parser.add_argument(
        "--world-size", type=int, default=4,
        help="World size to compare (default: %(default)s)")
    parser.add_argument(
        "--gnnflow-cache", type=str, default="LRUCache",
        help="GNNFlow cache name filter (default: %(default)s)")
    parser.add_argument(
        "--gnnflow-edge-cache-ratio", type=float, default=0.2,
        help="GNNFlow edge cache ratio filter (default: %(default)s)")
    parser.add_argument(
        "--gnnflow-node-cache-ratio", type=float, default=0.2,
        help="GNNFlow node cache ratio filter (default: %(default)s)")
    parser.add_argument(
        "--gnnflow-snapshot-time-window", type=float, default=0.0,
        help="GNNFlow snapshot time window filter (default: %(default)s)")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write comparison plots into (default: %(default)s)")
    parser.add_argument(
        "--formats", nargs="+", default=["pdf"],
        help="Output file formats to generate (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()

    disttgl_profiles_dir = DISTTGL_ROOT / "profiles"
    gnnflow_profiles_dir = args.gnnflow_root.resolve() / "profiles"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    disttgl_runs = collect_runs(disttgl_profiles_dir, "DistTGL")
    gnnflow_runs = collect_runs(gnnflow_profiles_dir, "GNNFlow")

    disttgl_selected = latest_run_per_batch(
        disttgl_runs,
        model=args.model,
        dataset=args.dataset,
        batch_sizes=args.batch_sizes,
        world_size=args.world_size,
    )
    gnnflow_selected = latest_run_per_batch(
        gnnflow_runs,
        model=args.model,
        dataset=args.dataset,
        batch_sizes=args.batch_sizes,
        world_size=args.world_size,
        cache=args.gnnflow_cache,
        edge_cache_ratio=args.gnnflow_edge_cache_ratio,
        node_cache_ratio=args.gnnflow_node_cache_ratio,
        snapshot_time_window=args.gnnflow_snapshot_time_window,
    )

    shared_batch_sizes = [
        batch_size for batch_size in args.batch_sizes
        if batch_size in disttgl_selected and batch_size in gnnflow_selected
    ]

    if len(shared_batch_sizes) == 0:
        raise SystemExit(
            "No matching TGN profiler runs were found for both systems at the "
            "requested batch sizes.")

    selected_runs = []
    for batch_size in shared_batch_sizes:
        selected_runs.append(gnnflow_selected[batch_size])
        selected_runs.append(disttgl_selected[batch_size])

    title_context = (
        f"{args.dataset} | model={args.model} | ws={args.world_size} | "
        f"GNNFlow cache={args.gnnflow_cache} e={args.gnnflow_edge_cache_ratio:g}/"
        f"n={args.gnnflow_node_cache_ratio:g}"
    )

    base_name = (
        f"system_compare_{args.model}_{args.dataset}_ws{args.world_size}_"
        f"bs{'-'.join(str(batch_size) for batch_size in shared_batch_sizes)}"
    )

    system_runs = {
        "GNNFlow": gnnflow_selected,
        "DistTGL": disttgl_selected,
    }

    for output_format in args.formats:
        plot_overview_comparison(
            shared_batch_sizes,
            system_runs,
            output_dir / f"{base_name}_overview.{output_format}",
            title_context)
        plot_stage_breakdown_comparison(
            shared_batch_sizes,
            system_runs,
            output_dir / f"{base_name}_stages.{output_format}",
            title_context)

    write_selected_runs_csv(
        selected_runs, output_dir / f"{base_name}_selected-runs.csv")

    print(f"Wrote comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
