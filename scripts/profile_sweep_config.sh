#!/bin/bash

# Example config file for scripts/run_profile_sweep.sh
#
# Usage:
#   ./scripts/run_profile_sweep.sh ./scripts/profile_sweep_config.sh

DATASETS=(REDDIT)

# DistTGL currently behaves like a TGN-style system in this repo, so use TGN
# as the profiler model label for cross-system comparisons.
PROFILE_MODEL_NAME="TGN"

BATCH_SIZES=(
  1024
  2048
  4096
  8192
  16384
  32768
)

NPROC_PER_NODE="4"
GROUP="1"
MINIBATCH_PARALLELISM="1"
NEG_SETS="32"
TRAIN_NEG_SAMPLES="1"
EVAL_NEG_SAMPLES="49"
OMP_NUM_THREADS="6"
SEED="0"
AUTO_GENERATE_MINIBATCHES="1"
FORCE_REGENERATE_MINIBATCHES="0"
BUILD_EXT_INPLACE="1"

PROFILE_ONLY="1"
PROFILE_WAIT="1"
PROFILE_WARMUP="1"
PROFILE_ACTIVE="6"
PROFILE_REPEAT="1"
PROFILE_ROW_LIMIT="50"
PROFILE_GPU_SAMPLE_INTERVAL="0.2"
PROFILE_RECORD_SHAPES="1"
PROFILE_WITH_STACK="0"
PROFILE_WITH_FLOPS="0"
PROFILE_EXPORT_MEMORY_TIMELINE="0"

PYTHON_BIN="python"
DRY_RUN="0"

# You can add extra arguments if needed, for example:
# EXTRA_ARGS=(--partial_eval)
EXTRA_ARGS=()
