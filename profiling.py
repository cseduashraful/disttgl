import json
import contextlib
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import GPUtil
import torch

try:
    from torch.profiler import (ProfilerActivity, profile as torch_profile,
                                record_function,
                                schedule as profiler_schedule,
                                tensorboard_trace_handler)
except ImportError:
    ProfilerActivity = None
    torch_profile = None
    profiler_schedule = None
    tensorboard_trace_handler = None

    def record_function(_name):
        return contextlib.nullcontext()


STANDARD_TRAIN_STAGE_NAMES = [
    "sampling",
    "feature_fetch",
    "memory_fetch",
    "memory_update",
    "memory_write_back",
    "model_forward",
    "loss_backward_optimizer",
]


def ensure_profiler_available():
    if torch_profile is None:
        raise RuntimeError(
            "torch.profiler is not available in the current torch installation")


def percentile(values: List[float], value: float) -> Optional[float]:
    if len(values) == 0:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * value
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def get_max_rss_bytes() -> Optional[int]:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system_name = os.uname().sysname.lower() if hasattr(os, "uname") else ""
    if system_name == "darwin":
        return int(usage)
    return int(usage) * 1024


def sanitize_run_component(value: Any) -> str:
    return ''.join(
        character if character.isalnum() or character in ('-', '_', '.')
        else '-'
        for character in str(value))


def maybe_synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def aggregate_metric(summaries: List[Dict[str, Any]],
                     path: List[str]) -> Dict[str, Optional[float]]:
    values: List[float] = []
    for summary in summaries:
        current: Any = summary
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            values.append(float(current))
    if len(values) == 0:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


class GpuStatsMonitor(threading.Thread):
    def __init__(self, cuda_device_index: int, sample_interval: float):
        super().__init__(daemon=True)
        self._sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._physical_gpu_index = self._resolve_physical_gpu_index(
            cuda_device_index)
        self.samples: List[Dict[str, float]] = []

    @staticmethod
    def _resolve_physical_gpu_index(cuda_device_index: int) -> int:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is None:
            return cuda_device_index

        mapped_devices = []
        for token in visible_devices.split(','):
            token = token.strip()
            if token == "" or token.startswith("GPU-"):
                return cuda_device_index
            try:
                mapped_devices.append(int(token))
            except ValueError:
                return cuda_device_index

        if 0 <= cuda_device_index < len(mapped_devices):
            return mapped_devices[cuda_device_index]
        return cuda_device_index

    def run(self):
        while not self._stop_event.is_set():
            try:
                gpu = next(
                    (gpu for gpu in GPUtil.getGPUs()
                     if gpu.id == self._physical_gpu_index),
                    None)
                if gpu is not None:
                    self.samples.append({
                        "timestamp": time.time(),
                        "load_pct": float(gpu.load) * 100.0,
                        "memory_util_pct": float(gpu.memoryUtil) * 100.0,
                        "memory_used_mb": float(gpu.memoryUsed),
                        "memory_total_mb": float(gpu.memoryTotal),
                    })
            except Exception:
                pass
            self._stop_event.wait(self._sample_interval)

    def stop(self):
        self._stop_event.set()

    def summarize(self) -> Dict[str, Optional[float]]:
        if len(self.samples) == 0:
            return {
                "sample_count": 0,
                "avg_load_pct": None,
                "max_load_pct": None,
                "avg_memory_util_pct": None,
                "max_memory_util_pct": None,
                "avg_memory_used_mb": None,
                "max_memory_used_mb": None,
            }

        load_values = [sample["load_pct"] for sample in self.samples]
        memory_util_values = [
            sample["memory_util_pct"] for sample in self.samples]
        memory_used_values = [
            sample["memory_used_mb"] for sample in self.samples]
        return {
            "sample_count": len(self.samples),
            "avg_load_pct": statistics.mean(load_values),
            "max_load_pct": max(load_values),
            "avg_memory_util_pct": statistics.mean(memory_util_values),
            "max_memory_util_pct": max(memory_util_values),
            "avg_memory_used_mb": statistics.mean(memory_used_values),
            "max_memory_used_mb": max(memory_used_values),
        }


class TrainingProfiler:
    def __init__(self, args, device: torch.device, batch_size: int,
                 setup_metrics: Dict[str, float], repo_root: Path):
        self.args = args
        self.device = device
        self.batch_size = batch_size
        self.setup_metrics = setup_metrics
        self.enabled = bool(args.profile)
        self.finished = False
        self.started = False
        self.total_steps_seen = 0
        self.step_records: List[Dict[str, Any]] = []
        self.profiler = None
        self.gpu_monitor = None
        self.repo_root = repo_root

        base_profile_dir = Path(args.profile_dir).resolve() \
            if args.profile_dir else repo_root / "profiles"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = "{}_{}_bs{}_ranks{}_{}".format(
            sanitize_run_component(args.profile_model_name),
            sanitize_run_component(args.data),
            batch_size,
            args.world_size,
            timestamp)
        self.output_dir = base_profile_dir / run_name
        self.rank_dir = self.output_dir / f"rank{args.rank}"
        self.trace_dir = self.rank_dir / "traces"
        self.summary_path = self.rank_dir / "summary.json"
        self.aggregate_summary_path = self.output_dir / "summary_all_ranks.json"

        cycle_length = args.profile_wait + args.profile_warmup + args.profile_active
        self.total_profile_steps = cycle_length * args.profile_repeat \
            if self.enabled else 0

        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _phase_for_step(self, step_index: int) -> str:
        cycle_length = self.args.profile_wait + self.args.profile_warmup + \
            self.args.profile_active
        if cycle_length <= 0 or step_index >= self.total_profile_steps:
            return "complete"
        offset = step_index % cycle_length
        if offset < self.args.profile_wait:
            return "wait"
        if offset < self.args.profile_wait + self.args.profile_warmup:
            return "warmup"
        return "active"

    def start(self):
        if not self.enabled or self.started:
            return

        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.gpu_monitor = GpuStatsMonitor(
                self.device.index if self.device.index is not None else 0,
                self.args.profile_gpu_sample_interval)
            self.gpu_monitor.start()

        self.profiler = torch_profile(
            activities=activities,
            schedule=profiler_schedule(
                wait=self.args.profile_wait,
                warmup=self.args.profile_warmup,
                active=self.args.profile_active,
                repeat=self.args.profile_repeat),
            on_trace_ready=tensorboard_trace_handler(
                str(self.trace_dir), worker_name=f"rank{self.args.rank}"),
            record_shapes=self.args.profile_record_shapes,
            profile_memory=True,
            with_stack=self.args.profile_with_stack,
            with_flops=self.args.profile_with_flops)
        self.profiler.start()
        self.started = True

    def begin_step(self, num_samples: int) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.finished:
            return None

        maybe_synchronize(self.device)
        state = {
            "step_index": self.total_steps_seen,
            "phase": self._phase_for_step(self.total_steps_seen),
            "started_at": time.perf_counter(),
            "num_samples": num_samples,
        }
        return state

    def end_step(self, state: Optional[Dict[str, Any]],
                 stage_durations: Dict[str, float]):
        if not self.enabled or self.finished or state is None:
            return

        maybe_synchronize(self.device)
        wall_time = time.perf_counter() - state["started_at"]
        step_record: Dict[str, Any] = {
            "step_index": int(state["step_index"]),
            "phase": state["phase"],
            "num_samples": int(state["num_samples"]),
            "wall_time_sec": float(wall_time),
            "throughput_samples_per_sec":
                float(state["num_samples"]) / wall_time if wall_time > 0 else None,
            "cpu_max_rss_bytes": get_max_rss_bytes(),
        }

        for stage_name in STANDARD_TRAIN_STAGE_NAMES:
            step_record[f"{stage_name}_sec"] = float(stage_durations.get(stage_name, 0.0))

        if self.device.type == "cuda":
            step_record["cuda_memory_allocated_end_bytes"] = int(
                torch.cuda.memory_allocated(self.device))
            step_record["cuda_memory_reserved_end_bytes"] = int(
                torch.cuda.memory_reserved(self.device))
            step_record["cuda_max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device))
            step_record["cuda_max_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device))

        self.step_records.append(step_record)
        self.profiler.step()
        self.total_steps_seen += 1

        if self.total_steps_seen >= self.total_profile_steps:
            self.finish()

    def is_complete(self) -> bool:
        return self.finished

    @staticmethod
    def _summarize_values(values: List[float]) -> Dict[str, Optional[float]]:
        numeric_values = [float(value) for value in values if value is not None]
        if len(numeric_values) == 0:
            return {
                "count": 0,
                "avg": None,
                "median": None,
                "min": None,
                "max": None,
                "p95": None,
            }
        return {
            "count": len(numeric_values),
            "avg": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "p95": percentile(numeric_values, 0.95),
        }

    def _build_summary(self) -> Dict[str, Any]:
        profiled_steps = [
            step for step in self.step_records if step["phase"] == "active"]
        if len(profiled_steps) == 0:
            profiled_steps = self.step_records

        summary: Dict[str, Any] = {
            "repo": "DistTGL",
            "model": self.args.profile_model_name,
            "dataset": self.args.data,
            "cache": "mailbox",
            "edge_cache_ratio": 0.0,
            "node_cache_ratio": 0.0,
            "snapshot_time_window": 0.0,
            "ingestion_batch_size": 0,
            "rank": self.args.rank,
            "world_size": self.args.world_size,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "profile_only": self.args.profile_only,
            "profile_schedule": {
                "wait": self.args.profile_wait,
                "warmup": self.args.profile_warmup,
                "active": self.args.profile_active,
                "repeat": self.args.profile_repeat,
                "total_profile_steps": self.total_profile_steps,
            },
            "setup_metrics_sec": self.setup_metrics,
            "steps_seen": self.total_steps_seen,
            "profiled_steps": len(profiled_steps),
            "step_time_sec": self._summarize_values(
                [step["wall_time_sec"] for step in profiled_steps]),
            "throughput_samples_per_sec": self._summarize_values([
                step["throughput_samples_per_sec"]
                for step in profiled_steps
                if step["throughput_samples_per_sec"] is not None
            ]),
            "stage_time_sec": {
                stage_name: self._summarize_values([
                    step[f"{stage_name}_sec"] for step in profiled_steps
                ])
                for stage_name in STANDARD_TRAIN_STAGE_NAMES
            },
            "cpu_memory": {
                "max_rss_bytes": max([
                    step["cpu_max_rss_bytes"]
                    for step in profiled_steps
                    if step.get("cpu_max_rss_bytes") is not None
                ], default=None),
            },
            "artifacts": {
                "rank_dir": str(self.rank_dir),
                "trace_dir": str(self.trace_dir),
            },
            "disttgl": {
                "group": self.args.group,
                "minibatch_parallelism": self.args.minibatch_parallelism,
                "neg_sets": self.args.neg_sets,
                "train_neg_samples": self.args.train_neg_samples,
            },
        }

        if self.device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            allocator_stats = torch.cuda.memory_stats(self.device)
            summary["cuda_memory"] = {
                "device_name": torch.cuda.get_device_name(self.device),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                "free_bytes_after_profile": int(free_bytes),
                "total_device_bytes": int(total_bytes),
                "alloc_retries": int(allocator_stats.get("num_alloc_retries", 0)),
                "ooms": int(allocator_stats.get("num_ooms", 0)),
                "active_peak_bytes": int(
                    allocator_stats.get("active_bytes.all.peak", 0)),
                "requested_peak_bytes": int(
                    allocator_stats.get("requested_bytes.all.peak", 0)),
            }
            summary["gpu_monitor"] = self.gpu_monitor.summarize() \
                if self.gpu_monitor is not None else {}
        else:
            summary["cuda_memory"] = {}
            summary["gpu_monitor"] = {}

        return summary

    def _write_profiler_tables(self):
        if self.profiler is None:
            return

        key_averages = self.profiler.key_averages()
        table_specs = [
            ("cpu_time_total", "key_averages_cpu_time.txt"),
            ("self_cpu_time_total", "key_averages_self_cpu_time.txt"),
            ("cuda_time_total", "key_averages_cuda_time.txt"),
            ("self_cuda_time_total", "key_averages_self_cuda_time.txt"),
            ("self_cpu_memory_usage", "key_averages_self_cpu_memory.txt"),
            ("self_cuda_memory_usage", "key_averages_self_cuda_memory.txt"),
        ]

        for sort_key, file_name in table_specs:
            try:
                table = key_averages.table(
                    sort_by=sort_key, row_limit=self.args.profile_row_limit)
            except Exception:
                continue
            with open(self.rank_dir / file_name, "w", encoding="utf-8") as handle:
                handle.write(table)

        if self.args.profile_with_stack and \
                hasattr(self.profiler, "export_stacks"):
            for metric in ("self_cpu_time_total", "self_cuda_time_total"):
                try:
                    self.profiler.export_stacks(
                        str(self.rank_dir / f"{metric}.txt"),
                        metric)
                except Exception:
                    continue

        if self.args.profile_export_memory_timeline and \
                hasattr(self.profiler, "export_memory_timeline") and \
                self.device.type == "cuda":
            output_path = self.rank_dir / "memory_timeline.html"
            try:
                self.profiler.export_memory_timeline(
                    str(output_path), device=str(self.device))
            except TypeError:
                try:
                    self.profiler.export_memory_timeline(str(output_path))
                except Exception:
                    pass
            except Exception:
                pass

    def finish(self):
        if not self.enabled or self.finished:
            return

        self.finished = True
        if self.profiler is not None:
            self.profiler.stop()
        if self.gpu_monitor is not None:
            self.gpu_monitor.stop()
            self.gpu_monitor.join(timeout=2.0)

        self._write_profiler_tables()
        summary = self._build_summary()
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        summaries: List[Optional[Dict[str, Any]]] = [None] * self.args.world_size
        torch.distributed.all_gather_object(summaries, summary)
        if self.args.rank == 0:
            rank_summaries = [item for item in summaries if item is not None]
            aggregate_summary = {
                "repo": "DistTGL",
                "model": self.args.profile_model_name,
                "dataset": self.args.data,
                "cache": "mailbox",
                "edge_cache_ratio": 0.0,
                "node_cache_ratio": 0.0,
                "snapshot_time_window": 0.0,
                "ingestion_batch_size": 0,
                "batch_size": self.batch_size,
                "world_size": self.args.world_size,
                "output_dir": str(self.output_dir),
                "setup_metrics_sec": {
                    name: aggregate_metric(rank_summaries, ["setup_metrics_sec", name])
                    for name in self.setup_metrics.keys()
                },
                "step_time_sec": aggregate_metric(
                    rank_summaries, ["step_time_sec", "avg"]),
                "stage_time_sec": {
                    stage_name: aggregate_metric(
                        rank_summaries, ["stage_time_sec", stage_name, "avg"])
                    for stage_name in STANDARD_TRAIN_STAGE_NAMES
                },
                "throughput_samples_per_sec": aggregate_metric(
                    rank_summaries, ["throughput_samples_per_sec", "avg"]),
                "peak_allocated_bytes": aggregate_metric(
                    rank_summaries, ["cuda_memory", "peak_allocated_bytes"]),
                "peak_reserved_bytes": aggregate_metric(
                    rank_summaries, ["cuda_memory", "peak_reserved_bytes"]),
                "gpu_load_pct": aggregate_metric(
                    rank_summaries, ["gpu_monitor", "avg_load_pct"]),
                "gpu_memory_util_pct": aggregate_metric(
                    rank_summaries, ["gpu_monitor", "avg_memory_util_pct"]),
                "gpu_memory_used_mb": aggregate_metric(
                    rank_summaries, ["gpu_monitor", "avg_memory_used_mb"]),
                "cpu_max_rss_bytes": aggregate_metric(
                    rank_summaries, ["cpu_memory", "max_rss_bytes"]),
                "ranks": rank_summaries,
            }
            with open(self.aggregate_summary_path, "w", encoding="utf-8") as handle:
                json.dump(aggregate_summary, handle, indent=2, sort_keys=True)
