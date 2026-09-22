#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RETENTION_CODE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
[[ -z "${MA_NUM_HOSTS:-}" ]] || fail "platform metadata detected; use run_mtp.sh on MTP"
export RETENTION_EXECUTION=local
export MINICONDA_PATH=/home/ma-user/work/dataset/Common_wl/miniconda3
load_settings
[[ "${CONDA_PREFIX:-}" == "$RETENTION_SIM_PREFIX" ]] || fail "locally activate RETENTION_SIM_PREFIX before starting this launcher"
# No MTP symlinks or conda init on the local debugging host.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$SCRIPT_DIR/dispatch.sh"
dispatch "$@"
