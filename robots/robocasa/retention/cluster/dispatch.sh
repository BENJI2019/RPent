#!/bin/bash
# Called only after the local/MTP environment has been prepared.
role_python() {
    local role="$1" prefix; shift
    case "$role" in
        simulator) prefix="$RETENTION_SIM_PREFIX" ;;
        openpi) prefix="$RETENTION_OPENPI_PREFIX" ;;
        qwen) prefix="$RETENTION_QWEN_PREFIX" ;;
        *) fail "unknown role: $role" ;;
    esac
    bash "$SCRIPT_DIR/python.sh" "$role" "$prefix" "$@"
}

config_for() {
    [[ "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || fail "experiment name must use letters, digits, '_' or '-'"
    CONFIG="$RETENTION_HOME/configs/$RETENTION_EXECUTION/$1.json"
}

check_config() {
    [[ -f "$CONFIG" ]] || fail "initialize the experiment first: $CONFIG"
    role_python simulator - "$CONFIG" <<'PY'
import os
import sys
from pathlib import Path
from robots.robocasa.retention.protocol import Experiment

experiment = Experiment.load(sys.argv[1])
expected = Path(os.environ["RETENTION_CODE_ROOT"]) / "vendor/openpi"
if Path(experiment.openpi_root).resolve() != expected.resolve():
    raise ValueError("config refers to another algorithm snapshot; initialize on this platform with a fresh output")
if not Path(experiment.output_root).resolve().is_relative_to(Path(os.environ["RETENTION_HOME"]).resolve()):
    raise ValueError("cluster outputs must stay under RETENTION_HOME on the model volume")
PY
}

dispatch() {
    local action="${1:-}" name preset mode gpu; shift || true
    case "$action" in
        python) role_python "$@" ;;
        init)
            preset="${1:?preset required}"; name="${2:?experiment name required}"; shift 2
            config_for "$name"
            role_python simulator -m robots.robocasa.retention --config "$CONFIG" init \
                --profile huawei --preset "$preset" --fsdp-devices "${MA_NUM_GPUS:-1}" \
                --output-root "$RETENTION_HOME/retention_outputs/$RETENTION_EXECUTION/$name" "$@"
            ;;
        cli)
            name="${1:?experiment name required}"; shift
            config_for "$name"; check_config
            role_python simulator -m robots.robocasa.retention --config "$CONFIG" "$@"
            ;;
        train)
            preset="${1:?preset required}"; name="${2:?experiment name required}"; shift 2
            [[ $# == 0 ]] || fail "train accepts PRESET NAME; customize with init before training"
            config_for "$name"
            if [[ ! -e "$CONFIG" ]]; then dispatch init "$preset" "$name"; fi
            dispatch cli "$name" doctor --stage train --runtime
            dispatch cli "$name" plan
            dispatch cli "$name" train --all --resume
            ;;
        planner)
            name="${1:?experiment name required}"; mode="${2:?planner mode required}"; gpu="${3:?physical GPU ordinal required}"
            [[ "$gpu" =~ ^[0-9]+$ ]] || fail "planner GPU must be a physical ordinal"
            config_for "$name"; check_config
            CUDA_VISIBLE_DEVICES="$gpu" role_python qwen -m robots.robocasa.retention.serve_planner --config "$CONFIG" --mode "$mode"
            ;;
        evaluate)
            name="${1:?experiment name required}"; mode="${2:?evaluation mode required}"; shift 2
            config_for "$name"; check_config
            dispatch cli "$name" plan
            role_python simulator -m robots.robocasa.retention.cluster_evaluate \
                --config "$CONFIG" --mode "$mode" --gpus "$CUDA_VISIBLE_DEVICES" "$@"
            ;;
        *) fail "usage: run_{local,mtp}.sh {init PRESET NAME [init options]|train PRESET NAME|cli NAME ARGS...|evaluate NAME MODE [--checkpoint base|all]|planner NAME MODE GPU|python ROLE ARGS...}" ;;
    esac
}
