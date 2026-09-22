#!/bin/bash
# Shared launcher functions. Sourcing this file alone has no filesystem effects.

fail() { printf 'retention: %s\n' "$*" >&2; exit 1; }

validate_mtp() {
    [[ "${MA_NUM_HOSTS:-}" == 1 ]] || fail "MTP requires MA_NUM_HOSTS=1; multi-host JAX is not implemented"
    [[ "${MA_NUM_GPUS:-}" =~ ^[1-9][0-9]*$ ]] || fail "missing/invalid MTP MA_NUM_GPUS; never defaults to 8"
    [[ "${VC_TASK_INDEX:-}" == 0 ]] || fail "MTP requires VC_TASK_INDEX=0"
    [[ -n "${VC_WORKER_HOSTS:-}" ]] || fail "MTP requires VC_WORKER_HOSTS"
    [[ "$RETENTION_CODE_ROOT/" == /opt/huawei/schedule-train/algorithm/* ]] || fail "run_mtp.sh must run from the submitted MTP algorithm snapshot"
}

ensure_link() {
    local target="$1" link="$2"
    [[ -d "$target" ]] || fail "shared mount is missing: $target"
    if [[ -e "$link" || -L "$link" ]]; then
        [[ -L "$link" && "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] || fail "refusing to replace existing path: $link"
    else
        mkdir -p "$(dirname "$link")"
        ln -s "$target" "$link"
    fi
}

mtp_mounts() {
    ensure_link /opt/huawei/dataset /opt/huawei/explorer-env/dataset
    ensure_link /opt/huawei/quoteModel /opt/huawei/explorer-env/quoteModel
    ensure_link /opt/huawei/dataset /home/ma-user/work/dataset
    ensure_link /opt/huawei/quoteModel /home/ma-user/work/model
}

configure_cuda() {
    local dataset="$1"
    export SETUPTOOLS_USE_DISTUTILS=local
    export CUDA_HOME="$dataset/Common_wl/pkgs_llb/cuda-12.8"
    export NCCL_HOME="$dataset/Common_wl/pkgs_llb/nccl"
    [[ -x "$CUDA_HOME/bin/nvcc" && -d "$NCCL_HOME/lib" ]] || fail "shared CUDA 12.8/NCCL is missing"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    export CUDAToolkit_ROOT="$CUDA_HOME" CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
    export CPATH="$NCCL_HOME/include:${CPATH:-}"
    export LIBRARY_PATH="$NCCL_HOME/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
    export DEEPSPEED_TIMEOUT=3000 NCCL_CONNECT_TIMEOUT=18000
    export NCCL_NET_GDR_LEVEL=0 NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
}

shared_path() {
    case "$1/" in
        /home/ma-user/work/dataset/*|/home/ma-user/work/model/*|/opt/huawei/dataset/*|/opt/huawei/quoteModel/*) ;;
        *) fail "path must be on a shared dataset/model volume: $1" ;;
    esac
    [[ "$1" != *'/../'* && "$1" != */.. ]] || fail "path must not contain '..': $1"
}

load_settings() {
    local settings="$RETENTION_CODE_ROOT/robots/robocasa/retention/cluster/cluster.env"
    [[ -f "$settings" ]] || fail "copy cluster.env.example to cluster.env and record your absolute Conda prefixes"
    source "$settings"
    local key
    for key in RETENTION_HOME RETENTION_DATASETS RETENTION_CHECKPOINT ROBOCASA_ASSETS_PATH RETENTION_SIM_PREFIX RETENTION_OPENPI_PREFIX RETENTION_QWEN_PREFIX RETENTION_TOOLS_PREFIX; do
        [[ -n "${!key:-}" ]] || fail "set $key in cluster.env"
        shared_path "${!key}"
    done
    # RETENTION_HOME must be a verified user-writable directory on shared storage.
    # Keep build products, tokenizer/model downloads and simulator images off code.
    export HF_HOME="$RETENTION_HOME/cache/huggingface" HF_HUB_CACHE="$RETENTION_HOME/cache/huggingface/hub"
    export XDG_CACHE_HOME="$RETENTION_HOME/cache/xdg" TORCH_HOME="$RETENTION_HOME/cache/torch"
    export OPENPI_DATA_HOME="$RETENTION_HOME/cache/openpi"
    export JAX_COMPILATION_CACHE_DIR="$RETENTION_HOME/cache/jax"
    export VLLM_CACHE_ROOT="$RETENTION_HOME/cache/vllm" TRITON_CACHE_DIR="$RETENTION_HOME/cache/triton"
    export UV_CACHE_DIR="$RETENTION_HOME/cache/uv" PIP_CACHE_DIR="$RETENTION_HOME/cache/pip"
    export CONDA_PKGS_DIRS="$RETENTION_HOME/cache/conda/pkgs"
    export TMPDIR="$RETENTION_HOME/tmp" WANDB_DIR="$RETENTION_HOME/wandb"
    export WANDB_CACHE_DIR="$RETENTION_HOME/cache/wandb" PYTHONDONTWRITEBYTECODE=1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
    mkdir -p "$TMPDIR" "$RETENTION_HOME/configs" "$RETENTION_HOME/env" "$WANDB_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
    # Ensure the submitted code wins over stale editable installs in shared envs.
    export PYTHONPATH="$RETENTION_CODE_ROOT"
}

strip_environment_paths() {
    local key part prefix kept resolved
    local -a parts prefixes=()
    for prefix in "$RETENTION_SIM_PREFIX" "$RETENTION_OPENPI_PREFIX" "$RETENTION_QWEN_PREFIX" "${RETENTION_TOOLS_PREFIX:-}" "${CONDA_PREFIX:-}" "${CUDNN_HOME:-}"; do
        [[ -n "$prefix" ]] || continue
        prefixes+=("$prefix")
        resolved="$(readlink -f "$prefix" || true)"
        [[ -z "$resolved" ]] || prefixes+=("$resolved")
    done
    for key in PATH LD_LIBRARY_PATH CPATH LIBRARY_PATH; do
        kept=""
        IFS=: read -r -a parts <<< "${!key:-}"
        for part in "${parts[@]}"; do
            [[ -n "$part" ]] || continue
            for prefix in "${prefixes[@]}"; do
                if [[ -n "$prefix" && ( "$part" == "$prefix" || "$part" == "$prefix/"* ) ]]; then
                    part=""; break
                fi
            done
            [[ -z "$part" ]] || kept="${kept:+$kept:}$part"
        done
        export "$key=$kept"
    done
    unset CUDNN_HOME
}

activate_role() {
    local prefix="$1" candidate cuda_root="${CUDA_HOME:-}" nccl_root="${NCCL_HOME:-}"
    shared_path "$prefix"
    [[ -x "$prefix/bin/python" && -d "$prefix/conda-meta" ]] || fail "missing Conda environment: $prefix"
    strip_environment_paths
    source "$MINICONDA_PATH/etc/profile.d/conda.sh"
    conda activate "$prefix"
    # Some activation hooks restore the previous role's library paths.
    strip_environment_paths
    export PATH="$prefix/bin:$PATH" LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"
    if [[ -n "$nccl_root" ]]; then
        export NCCL_HOME="$nccl_root"
        export CPATH="$NCCL_HOME/include:${CPATH:-}" LIBRARY_PATH="$NCCL_HOME/lib:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
    fi
    if [[ -n "$cuda_root" ]]; then
        export CUDA_HOME="$cuda_root" CUDACXX="$cuda_root/bin/nvcc"
        export CUDAToolkit_ROOT="$cuda_root" CUDA_TOOLKIT_ROOT_DIR="$cuda_root"
        export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    fi
    for candidate in "$prefix"/lib/python*/site-packages/nvidia/cudnn; do
        if [[ -d "$candidate/lib" ]]; then
            export CUDNN_HOME="$candidate"
            export CPATH="$candidate/include:${CPATH:-}" LIBRARY_PATH="$candidate/lib:${LIBRARY_PATH:-}"
            export LD_LIBRARY_PATH="$candidate/lib:$LD_LIBRARY_PATH"
            break
        fi
    done
}
