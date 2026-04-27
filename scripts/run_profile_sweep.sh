#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_profile_sweep.sh [optional-config-file]

Description:
  Run a profiling sweep across multiple dataset / batch-size combinations for
  DistTGL. Edit the USER CONFIG block in this script directly, or pass a shell
  config file that overrides the variables.

Example:
  ./scripts/run_profile_sweep.sh
  ./scripts/run_profile_sweep.sh ./scripts/profile_sweep_config.sh
EOF
}

if [[ $# -gt 1 ]]; then
    usage
    exit 1
fi

#
# USER CONFIG
#

DATASETS=(REDDIT)
BATCH_SIZES=(3200 6400)

PROFILE_MODEL_NAME="TGN"
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
LOG_DIR="${REPO_ROOT}/profiles/logs"
PROFILE_DIR_ROOT="${REPO_ROOT}/profiles"
DRY_RUN="0"

# Extra arguments appended after the explicit profiler arguments.
EXTRA_ARGS=()

if [[ $# -eq 1 ]]; then
    # shellcheck source=/dev/null
    source "$1"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${PROFILE_DIR_ROOT}"

expected_train_minibatch_dir() {
    local dataset="$1"
    local batch_size="$2"
    if [[ "${MINIBATCH_PARALLELISM}" -gt 1 ]]; then
        printf '%s/minibatches/%s_%s_bs%s_%s_%s_%s/' \
            "${REPO_ROOT}" "${MINIBATCH_PARALLELISM}" "${dataset}" "${batch_size}" \
            "${TRAIN_NEG_SAMPLES}" "${EVAL_NEG_SAMPLES}" "${NEG_SETS}"
    else
        printf '%s/minibatches/%s_bs%s_%s_%s_%s/' \
            "${REPO_ROOT}" "${dataset}" "${batch_size}" \
            "${TRAIN_NEG_SAMPLES}" "${EVAL_NEG_SAMPLES}" "${NEG_SETS}"
    fi
}

expected_eval_minibatch_dir() {
    local dataset="$1"
    local batch_size="$2"
    printf '%s/minibatches/%s_bs%s_%s_eval/' \
        "${REPO_ROOT}" "${dataset}" "${batch_size}" "${EVAL_NEG_SAMPLES}"
}

build_sampler_extension() {
    if [[ "${BUILD_EXT_INPLACE}" != "1" ]]; then
        return 0
    fi
    printf '\nBuilding DistTGL sampler extension in-place\n'
    (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" setup.py build_ext --inplace
    )
}

generate_minibatches() {
    local dataset="$1"
    local batch_size="$2"
    printf '\nGenerating DistTGL minibatches for dataset=%s batch_size=%s\n' \
        "${dataset}" "${batch_size}"
    (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" gen_minibatch.py \
            --data "${dataset}" \
            --gen_eval \
            --minibatch_parallelism "${MINIBATCH_PARALLELISM}" \
            --train_neg_samples "${TRAIN_NEG_SAMPLES}" \
            --neg_sets "${NEG_SETS}" \
            --batchsize "${batch_size}"
    )
}

require_minibatches() {
    local dataset="$1"
    local batch_size="$2"
    local stats_path="${REPO_ROOT}/minibatches/${dataset}_stats.pkl"
    local train_dir
    local eval_dir
    train_dir="$(expected_train_minibatch_dir "${dataset}" "${batch_size}")"
    eval_dir="$(expected_eval_minibatch_dir "${dataset}" "${batch_size}")"

    if [[ "${FORCE_REGENERATE_MINIBATCHES}" == "1" ]]; then
        if [[ "${AUTO_GENERATE_MINIBATCHES}" == "1" ]]; then
            generate_minibatches "${dataset}" "${batch_size}"
        else
            printf '\nFORCE_REGENERATE_MINIBATCHES=1 but AUTO_GENERATE_MINIBATCHES=0 for dataset=%s batch_size=%s.\n' \
                "${dataset}" "${batch_size}" >&2
            exit 1
        fi
    fi

    if [[ -f "${stats_path}" && -d "${train_dir}" && \
          -f "${train_dir}/train_pos_0.pkl" && \
          -d "${eval_dir}" && -f "${eval_dir}/val_pos_0.pkl" ]]; then
        return 0
    fi

    if [[ "${AUTO_GENERATE_MINIBATCHES}" == "1" ]]; then
        generate_minibatches "${dataset}" "${batch_size}"
        if [[ -f "${stats_path}" && -d "${train_dir}" && \
              -f "${train_dir}/train_pos_0.pkl" && \
              -d "${eval_dir}" && -f "${eval_dir}/val_pos_0.pkl" ]]; then
            return 0
        fi
        printf '\nAutomatic minibatch generation finished but expected artifacts are still missing.\n' >&2
    fi

    printf '\nMissing DistTGL minibatches for dataset=%s batch_size=%s.\n' \
        "${dataset}" "${batch_size}" >&2
    printf 'Expected artifacts:\n' >&2
    printf '  %s\n' "${stats_path}" >&2
    printf '  %s\n' "${train_dir}" >&2
    printf '  %s\n' "${eval_dir}" >&2
    printf '\nGenerate them first with:\n' >&2
    printf '  cd %q\n' "${REPO_ROOT}" >&2
    printf '  %q setup.py build_ext --inplace\n' "${PYTHON_BIN}" >&2
    printf '  %q gen_minibatch.py --data %q --gen_eval --minibatch_parallelism %q --train_neg_samples %q --neg_sets %q --batchsize %q\n' \
        "${PYTHON_BIN}" "${dataset}" "${MINIBATCH_PARALLELISM}" \
        "${TRAIN_NEG_SAMPLES}" "${NEG_SETS}" "${batch_size}" >&2
    exit 1
}

printf 'Profiling sweep configuration\n'
printf '  datasets: %s\n' "${DATASETS[*]}"
printf '  batch sizes: %s\n' "${BATCH_SIZES[*]}"
printf '  profile model name: %s\n' "${PROFILE_MODEL_NAME}"
printf '  world size: %s\n' "${NPROC_PER_NODE}"
printf '  group: %s\n' "${GROUP}"
printf '  minibatch parallelism: %s\n' "${MINIBATCH_PARALLELISM}"
printf '  eval neg samples: %s\n' "${EVAL_NEG_SAMPLES}"
printf '  auto-generate minibatches: %s\n' "${AUTO_GENERATE_MINIBATCHES}"
printf '  force-regenerate minibatches: %s\n' "${FORCE_REGENERATE_MINIBATCHES}"
printf '  profile-only: %s\n' "${PROFILE_ONLY}"
printf '  profile dir: %s\n' "${PROFILE_DIR_ROOT}"
printf '  log dir: %s\n' "${LOG_DIR}"

if [[ "${AUTO_GENERATE_MINIBATCHES}" == "1" && "${DRY_RUN}" != "1" ]]; then
    build_sampler_extension
fi

for dataset in "${DATASETS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        require_minibatches "${dataset}" "${batch_size}"
        run_name="profile_${PROFILE_MODEL_NAME}_${dataset}_bs${batch_size}_group${GROUP}_mb${MINIBATCH_PARALLELISM}_ws${NPROC_PER_NODE}"
        log_path="${LOG_DIR}/${run_name}.log"

        common_args=(
            train.py
            --data "${dataset}"
            --group "${GROUP}"
            --minibatch_parallelism "${MINIBATCH_PARALLELISM}"
            --neg_sets "${NEG_SETS}"
            --train_neg_samples "${TRAIN_NEG_SAMPLES}"
            --omp_num_threads "${OMP_NUM_THREADS}"
            --seed "${SEED}"
            --batchsize "${batch_size}"
            --profile
            --profile-model-name "${PROFILE_MODEL_NAME}"
            --profile-dir "${PROFILE_DIR_ROOT}"
            --profile-wait "${PROFILE_WAIT}"
            --profile-warmup "${PROFILE_WARMUP}"
            --profile-active "${PROFILE_ACTIVE}"
            --profile-repeat "${PROFILE_REPEAT}"
            --profile-row-limit "${PROFILE_ROW_LIMIT}"
            --profile-gpu-sample-interval "${PROFILE_GPU_SAMPLE_INTERVAL}"
        )

        if [[ "${PROFILE_ONLY}" == "1" ]]; then
            common_args+=(--profile-only)
        fi

        if [[ "${PROFILE_RECORD_SHAPES}" == "0" ]]; then
            common_args+=(--no-profile-record-shapes)
        fi

        if [[ "${PROFILE_WITH_STACK}" == "1" ]]; then
            common_args+=(--profile-with-stack)
        fi

        if [[ "${PROFILE_WITH_FLOPS}" == "1" ]]; then
            common_args+=(--profile-with-flops)
        fi

        if [[ "${PROFILE_EXPORT_MEMORY_TIMELINE}" == "1" ]]; then
            common_args+=(--profile-export-memory-timeline)
        fi

        common_args+=("${EXTRA_ARGS[@]}")

        if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
            cmd=(
                torchrun
                --nnodes=1
                --nproc_per_node="${NPROC_PER_NODE}"
                --standalone
                "${common_args[@]}"
            )
        else
            cmd=(
                "${PYTHON_BIN}"
                "${common_args[@]}"
            )
        fi

        printf '\n=== %s ===\n' "${run_name}"
        printf 'log: %s\n' "${log_path}"
        printf '%q ' "${cmd[@]}"
        printf '\n'

        if [[ "${DRY_RUN}" == "1" ]]; then
            continue
        fi

        (
            cd "${REPO_ROOT}"
            OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${cmd[@]}"
        ) >"${log_path}" 2>&1
    done
done
