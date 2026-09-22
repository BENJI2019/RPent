#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
: "${RETENTION_CODE_ROOT:?Use run_local.sh or run_mtp.sh}"
: "${RETENTION_EXECUTION:?Use run_local.sh or run_mtp.sh}"
role="${1:?role required}"; prefix="${2:?absolute Conda prefix required}"; shift 2
case "$role" in simulator|openpi|qwen) ;; *) fail "invalid Python role: $role" ;; esac
activate_role "$prefix"
export PYTHONPATH="$RETENTION_CODE_ROOT"
if [[ "$role" == openpi ]]; then
    export PYTHONPATH="$RETENTION_CODE_ROOT/vendor/openpi/src:$RETENTION_CODE_ROOT/vendor/openpi/packages/openpi-client/src:$PYTHONPATH"
fi
if [[ "$role" == qwen ]]; then export VLLM_WORKER_MULTIPROCESS_METHOD=spawn; fi
export CUDA_CACHE_PATH="$RETENTION_HOME/cache/cuda/$role"
cd "$RETENTION_HOME"
exec "$prefix/bin/python" "$@"
