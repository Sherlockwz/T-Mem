#!/usr/bin/env bash
# ============================================================
# T_mem · LoCoMo QA + judge (requires an existing memory library).
#
# Consumes build_memory.sh outputs (search_results.json + personas) and runs:
#   Step 0  T_mem.io.truncate_search_results → trims stage6 wide recall (24 scene + 40 item)
#                                     down to final_keep_scene / final_keep_item
#                                     in-place. The trim K is read from
#                                     T_mem/config.py :: ExperimentConfig.retrieval_config
#                                     (single source of truth; default 5 scene + 15 item
#                                     == paper-final M1_N15 cell). Override via env:
#                                     T_MEM_FINAL_KEEP_SCENE / T_MEM_FINAL_KEEP_ITEM.
#   Step 1  stage8_qa_locomo      → responses.json
#   Step 2  predictions_adapter   → predictions.json
#   Step 3  convert_format        → locomo_responses.json
#   Step 4  memos_judge           → judge_results.json
#
# Usage:
#   bash scripts/eval_locomo.sh --resume <experiment_dir> [--judge-num-runs 3]
#
# Model ids are declared in T_mem/config.py :: MODELS (not via shell).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LOCOMO_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/locomo/data/locomo10.json"

EXP_DIR=""
LOCOMO_FILE="${T_MEM_LOCOMO_FILE:-$LOCOMO_FILE_DEFAULT}"
JUDGE_NUM_RUNS="${T_MEM_JUDGE_NUM_RUNS:-3}"
JUDGE_WORKERS="${T_MEM_JUDGE_WORKERS:-10}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)           EXP_DIR="$2";          shift 2;;
        --locomo-file)      LOCOMO_FILE="$2";      shift 2;;
        --judge-num-runs)   JUDGE_NUM_RUNS="$2";   shift 2;;
        --judge-workers)    JUDGE_WORKERS="$2";    shift 2;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0;;
        *)
            echo "[eval_locomo] unknown arg: $1" >&2
            exit 2;;
    esac
done

# ---------------- Preflight ----------------
if [[ -z "$EXP_DIR" ]]; then
    echo "[eval_locomo] FATAL: --resume <experiment_dir> is required" >&2
    echo "  Run scripts/build_memory.sh first to produce the memory library." >&2
    exit 2
fi
EXP_DIR="$(cd "$EXP_DIR" && pwd)"
if [[ ! -f "$EXP_DIR/search_results.json" ]]; then
    echo "[eval_locomo] FATAL: $EXP_DIR/search_results.json missing (stage 6 not done?)" >&2
    exit 2
fi
if [[ ! -f "$LOCOMO_FILE" ]]; then
    echo "[eval_locomo] FATAL: $LOCOMO_FILE missing" >&2
    exit 2
fi

PERSONA_STORE_ROOT="$EXP_DIR/personas"
LOG_DIR="$EXP_DIR/logs"
LOG_FILE="$LOG_DIR/eval_locomo.log"
mkdir -p "$LOG_DIR"

# ---------------- Env ----------------
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export T_MEM_FAILURE_LOG="${T_MEM_FAILURE_LOG:-$LOG_DIR/json_failures.jsonl}"
# Experiment env consistent with build_memory.sh so stage8_qa_locomo reads the same config.
export T_MEM_EXPERIMENT_NAME="$(basename "$EXP_DIR")"
export T_MEM_RESULTS_DIR="$(dirname "$EXP_DIR")"
export T_MEM_DATA_FILE="$LOCOMO_FILE"
if [[ -d "$PERSONA_STORE_ROOT" ]]; then
    export T_MEM_PERSONA_STORE_ROOT="$PERSONA_STORE_ROOT"
fi
# Pick up L2L3 per-QA top-K if stage 5 produced it.
if [[ -f "$EXP_DIR/l2l3_topk_per_qa.json" ]]; then
    export L2L3_ASSOC_TOPK_JSON="$EXP_DIR/l2l3_topk_per_qa.json"
fi

cd "$PROJECT_ROOT"

banner() {
    echo ""                                                   | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"                  | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
}

run_step() {
    local name="$1"; shift
    banner "START: $name"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        banner "FAILED: $name (exit=$rc)"
        exit "$rc"
    fi
    banner "DONE : $name"
}

banner "T_mem · LoCoMo QA+Judge"
echo "EXP_DIR            = $EXP_DIR"                          | tee -a "$LOG_FILE"
echo "LOCOMO_FILE        = $LOCOMO_FILE"                      | tee -a "$LOG_FILE"
echo "PERSONA_STORE_ROOT = ${T_MEM_PERSONA_STORE_ROOT:-<disabled>}" | tee -a "$LOG_FILE"
echo "JUDGE_NUM_RUNS     = $JUDGE_NUM_RUNS"                   | tee -a "$LOG_FILE"
echo "JUDGE_WORKERS      = $JUDGE_WORKERS"                    | tee -a "$LOG_FILE"
echo "Model ids          = T_mem/config.py :: MODELS"         | tee -a "$LOG_FILE"

# ---- Step 0/4: trim stage6 wide recall to final_keep_scene / final_keep_item ----
# stage6 emits a wide-recall master (24 scene + 40 item by default, see
# T_mem/config.py :: retrieval_config). The trimmer cuts that down to the
# K that actually enters the QA prompt; the K is read from the same config
# so this shell script holds NO magic numbers. Override via env, e.g.:
#   T_MEM_FINAL_KEEP_SCENE=10 T_MEM_FINAL_KEEP_ITEM=20 bash scripts/eval_locomo.sh ...
run_step "Step 0/4 trim search_results -> final_keep_* from T_mem/config.py (in-place)" \
    python3 -u -m T_mem.io.truncate_search_results \
        --src-dir    "$EXP_DIR" \
        --dst-dir    "$EXP_DIR"

# ---- Step 1/4: stage8 QA ----
run_step "Step 1/4 stage8 QA" \
    python3 -u -m T_mem.main.stage8_qa_locomo \
        --memory-dir "$EXP_DIR" \
        --out-dir    "$EXP_DIR"

# ---- Step 2/4: predictions_adapter ----
run_step "Step 2/4 predictions_adapter" \
    python3 -u -m T_mem.io.predictions_adapter \
        --locomo-file    "$LOCOMO_FILE" \
        --responses-file "$EXP_DIR/responses.json" \
        --output-file    "$EXP_DIR/predictions.json"

# ---- Step 3/4: convert_format (predictions.json -> locomo_responses.json) ----
run_step "Step 3/4 convert_format (judge input)" \
    python3 -u "$PROJECT_ROOT/benchmark_eval/locomo/runner/convert_format.py" \
        --predictions "$EXP_DIR/predictions.json" \
        --locomo      "$LOCOMO_FILE" \
        --output      "$EXP_DIR/locomo_responses.json"

# ---- Step 4/4: memos_judge (LoCoMo official) ----
run_step "Step 4/4 memos_judge (LoCoMo official)" \
    python3 -u "$PROJECT_ROOT/benchmark_eval/locomo/runner/memos_judge.py" \
        --input    "$EXP_DIR/locomo_responses.json" \
        --output   "$EXP_DIR/judge_results.json" \
        --num-runs "$JUDGE_NUM_RUNS" \
        --workers  "$JUDGE_WORKERS"

banner "ALL DONE"
echo "responses.json        = $EXP_DIR/responses.json"         | tee -a "$LOG_FILE"
echo "predictions.json      = $EXP_DIR/predictions.json"       | tee -a "$LOG_FILE"
echo "locomo_responses.json = $EXP_DIR/locomo_responses.json"  | tee -a "$LOG_FILE"
echo "judge_results.json    = $EXP_DIR/judge_results.json"     | tee -a "$LOG_FILE"
echo "Log                   = $LOG_FILE"                       | tee -a "$LOG_FILE"
