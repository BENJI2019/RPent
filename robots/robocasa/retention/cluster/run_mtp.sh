#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RETENTION_CODE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
# Reject accidental local execution BEFORE touching mounts or starting conda.
validate_mtp
export RETENTION_EXECUTION=mtp
mtp_mounts
configure_cuda /opt/huawei/dataset
export MINICONDA_PATH=/opt/huawei/dataset/Common_wl/miniconda3
export PATH="$MINICONDA_PATH/bin:$PATH"
"$MINICONDA_PATH/bin/conda" init bash
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
load_settings
activate_role "$RETENTION_SIM_PREFIX"
printf 'MTP hosts=%s GPUs=%s rank=%s workers=%s code=%s\n' "$MA_NUM_HOSTS" "$MA_NUM_GPUS" "$VC_TASK_INDEX" "$VC_WORKER_HOSTS" "$RETENTION_CODE_ROOT"
# This trainer uses one JAX process across local GPUs, without torchrun/Ray.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=0
    for ((i=1; i<MA_NUM_GPUS; i++)); do CUDA_VISIBLE_DEVICES+=",$i"; done
    export CUDA_VISIBLE_DEVICES
fi
source "$SCRIPT_DIR/dispatch.sh"
dispatch "$@"
