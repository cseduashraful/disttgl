#!/bin/bash

# Example config file for scripts/run_profile_sweep.sh
#
# Usage:
#   ./scripts/run_profile_sweep.sh ./scripts/profile_sweep_config.sh

DATASETS=(REDDIT)

# DistTGL trains a single architecture in this repo. This label is stored in
# profiler summaries so downstream plots can distinguish these runs cleanly.
PROFILE_MODEL_NAME="DistTGL"

BATCH_SIZES=(
  1600
  3200
  6400
  12800
  25600
)

NPROC_PER_NODE="4"
GROUP="1"
MINIBATCH_PARALLELISM="1"
NEG_SETS="32"
TRAIN_NEG_SAMPLES="1"
OMP_NUM_THREADS="6"
SEED="0"

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
