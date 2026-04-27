# DistTGL: Distributed Memory-based Temporal Graph Neural Network Training

## Overview

This repo is the open-sourced code for our work *DistTGL: Distributed Memory-based Temporal Graph Neural Network Training*.

## Requirements
- python >= 3.8.13
- pytorch >= 1.11.0
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.8.2
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

## Dataset
Download the dataset using the `down.sh` script. Note that we do not release the Flights dataset due to license restriction. You can download the Flight dataset directly from [this](https://github.com/fpour/DGB) link.

## Mini-batch Preparation

DistTGL pre-compute mini-batches before training. To ensure fast mini-batch loading from disk, please store the mini-batches in a fast SSD. In the paper, we use RAID0 array of two NVMe SSDs.

We first compile the sampler from [TGL](https://github.com/amazon-science/tgl) by
> python setup.py build_ext --inplace

Then generate the mini-batches using
> python gen_minibatch.py --data \<DatasetName> --gen_eval --minibatch_parallelism \<NumberofMinibatchParallelism>

where `<NumberofMinibatchParallelism>` is the `i` in `(i x j x k)` in the paper.

## Run

On each machine, execute
> torchrun --nnodes=\<NumberofMachines> --nproc_per_node=\<NumberofGPUPerMachine> --rdzv_id=\<JobID> --rdzv_backend=c10d --rdzv_endpoint=\<HostNodeIPAddress>:\<HostNodePort> train5.py --data \<DatsetName> --group \<NumberofGroupParallelism>

where `<NumberofGroupParallelism>` is the `k` in `(i x j x k)` in the paper.

## Profiling

The `profile` branch includes richer profiling hooks in `train.py` that export:
- per-rank TensorBoard traces
- per-rank `summary.json`
- aggregate `summary_all_ranks.json`
- operator summary tables for CPU time, CUDA time, and memory

Profiler outputs are written under `profiles/<run>/`.

Example profiling run:

> torchrun --nnodes=1 --nproc_per_node=4 --standalone train.py --data REDDIT --group 1 --batchsize 3200 --profile --profile-only --profile-wait 1 --profile-warmup 1 --profile-active 6 --profile-repeat 1

Before profiling or training, generate the precomputed minibatches for the
same batch size you plan to use:

> python setup.py build_ext --inplace
> python gen_minibatch.py --data REDDIT --gen_eval --minibatch_parallelism 1 --batchsize 3200

Useful profiling arguments:
- `--profile-only`: stop after enough training steps to collect profiler data
- `--profile-dir <path>`: override the output directory for profiler artifacts
- `--profile-model-name <name>`: label stored in profiler summaries, default `TGN`
- `--profile-wait`, `--profile-warmup`, `--profile-active`, `--profile-repeat`: profiler schedule controls
- `--profile-with-stack`, `--profile-with-flops`, `--profile-export-memory-timeline`: enable extra profiler outputs

The generated summaries track the same headline metrics as the GNNFlow profiler workflow:
- step time
- throughput
- peak allocated / reserved GPU memory
- average GPU load and GPU memory utilization
- average GPU memory used
- peak process RSS
- stage-level training timings

### Profiling sweep launcher

To mirror the GNNFlow workflow, this branch also includes:
- `scripts/run_profile_sweep.sh`
- `scripts/profile_sweep_config.sh`

Run the checked-in sweep config with:

> ./scripts/run_profile_sweep.sh ./scripts/profile_sweep_config.sh

The sweep script now checks for batch-size-specific minibatch artifacts before
launching each run.

By default it will also:
- run `python setup.py build_ext --inplace` once
- run `python gen_minibatch.py ... --batchsize <value>` automatically for each
  batch size in the config when artifacts are missing

You can control that behavior from `scripts/profile_sweep_config.sh` with:
- `AUTO_GENERATE_MINIBATCHES`
- `FORCE_REGENERATE_MINIBATCHES`
- `BUILD_EXT_INPLACE`

Edit `scripts/profile_sweep_config.sh` to change:
- datasets
- batch sizes
- profiler schedule values
- world size / group / minibatch parallelism
- the profiler label stored in summaries via `PROFILE_MODEL_NAME` (default `TGN`)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
