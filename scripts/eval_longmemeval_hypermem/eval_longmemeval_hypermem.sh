#!/usr/bin/env bash
# ============================================================
# T_mem · LongMemEval QA + judge (HyperMem protocol plugin).
#
# Consumes build_memory.sh --mode lme outputs (search_results.json + personas
# + stitched_lme.json) and runs:
#   Step -1 T_mem.main.stage6_retrieval → re-run stage6 without triggers
#            (default; skipped when --enable-triggers is passed)
#   Step 0  T_mem.io.truncate_search_results → trims stage6 wide recall
#   Step 1  qa_hypermem.py  → HyperMem 7-step CoT + FINAL ANSWER extraction
#                                → responses_hypermem.json + hypothesis_hypermem.jsonl
#   Step 2  judge_hypermem.py → HyperMem single CORRECT/WRONG template
#                                 + 3-run JSON judge (gpt-4o-mini)
#                                 → *.eval-results-gpt-4o-mini
#                                 + *.eval-results-gpt-4o-mini.metrics.json
#                                   (Overall mean±std + 6-qtype breakdown)
#
# 差异对照默认 eval_longmemeval.sh：
#   - QA prompt: HyperMem 7-step CoT（而非 LongMemEval 官方 con reader）
#   - QA 后处理: 提取 FINAL ANSWER（而非原样输出）
#   - Judge prompt: 通用 CORRECT/WRONG（而非 5+1 qtype 特化模板）
#   - Judge 解析: JSON label（而非 "yes" in response）
#   - Judge 模型: gpt-4o-mini（而非 gpt-4o）
#   - Judge 次数: 3（而非 1）
#
# Usage:
#   bash scripts/eval_longmemeval_hypermem/eval_longmemeval_hypermem.sh --resume <experiment_dir>
#   bash scripts/eval_longmemeval_hypermem/eval_longmemeval_hypermem.sh --resume <experiment_dir> --enable-triggers
#   bash scripts/eval_longmemeval_hypermem/eval_longmemeval_hypermem.sh --resume <experiment_dir> --judge-model gpt-4o --num-runs 5
#   bash scripts/eval_longmemeval_hypermem/eval_longmemeval_hypermem.sh --skip-truncate --resume <experiment_dir>
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LME_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/longmemeval/data/longmemeval_s_cleaned.json"

EXP_DIR=""
LME_FILE="${T_MEM_LME_FILE:-$LME_FILE_DEFAULT}"
JUDGE_MODEL="${T_MEM_LME_JUDGE_MODEL:-gpt-4o-mini}"
NUM_RUNS="${T_MEM_LME_NUM_RUNS:-3}"
JUDGE_CONCURRENCY="${T_MEM_LME_JUDGE_CONCURRENCY:-16}"
QA_CONCURRENCY="${T_MEM_QA_CONCURRENCY:-14}"
QA_MODEL="${T_MEM_LME_QA_MODEL:-gpt-4o-mini}"
SKIP_TRUNCATE=0
SKIP_QA=0
SKIP_JUDGE=0
ENABLE_TRIGGERS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)             EXP_DIR="$2";             shift 2;;
        --lme-file)           LME_FILE="$2";            shift 2;;
        --qa-model)           QA_MODEL="$2";            shift 2;;
        --qa-concurrency)     QA_CONCURRENCY="$2";      shift 2;;
        --judge-model)        JUDGE_MODEL="$2";         shift 2;;
        --num-runs)           NUM_RUNS="$2";            shift 2;;
        --judge-concurrency)  JUDGE_CONCURRENCY="$2";   shift 2;;
        --skip-truncate)      SKIP_TRUNCATE=1;          shift;;
        --skip-qa)            SKIP_QA=1;                shift;;
        --skip-judge)         SKIP_JUDGE=1;             shift;;
        --enable-triggers)    ENABLE_TRIGGERS=1;         shift;;
        -h|--help)
            sed -n '2,48p' "$0"
            exit 0;;
        *)
            echo "[eval_longmemeval_hypermem] unknown arg: $1" >&2
            exit 2;;
    esac
done

# ---------------- Preflight ----------------
if [[ -z "$EXP_DIR" ]]; then
    echo "[eval_longmemeval_hypermem] FATAL: --resume <experiment_dir> is required" >&2
    echo "  Run scripts/build_memory.sh --mode lme first to produce the memory library." >&2
    exit 2
fi
EXP_DIR="$(cd "$EXP_DIR" && pwd)"

if [[ ! -f "$EXP_DIR/search_results.json" ]]; then
    echo "[eval_longmemeval_hypermem] FATAL: $EXP_DIR/search_results.json missing (stage 6 not done?)" >&2
    exit 2
fi

STITCHED_FILE="$EXP_DIR/data/stitched_lme.json"
if [[ ! -f "$STITCHED_FILE" ]]; then
    echo "[eval_longmemeval_hypermem] FATAL: $STITCHED_FILE missing (stage 0 lme stitch not done?)" >&2
    exit 2
fi

if [[ ! -f "$LME_FILE" ]]; then
    echo "[eval_longmemeval_hypermem] FATAL: $LME_FILE missing (LongMemEval reference data)" >&2
    exit 2
fi

PERSONA_STORE_ROOT="$EXP_DIR/personas"
LOG_DIR="$EXP_DIR/logs"
LOG_FILE="$LOG_DIR/eval_longmemeval_hypermem.log"
mkdir -p "$LOG_DIR"

# ---------------- Env ----------------
export PYTHONPATH="$PROJECT_ROOT:$SCRIPT_DIR:${PYTHONPATH:-}"
export T_MEM_FAILURE_LOG="${T_MEM_FAILURE_LOG:-$LOG_DIR/json_failures.jsonl}"
export T_MEM_EXPERIMENT_NAME="$(basename "$EXP_DIR")"
export T_MEM_RESULTS_DIR="$(dirname "$EXP_DIR")"
export T_MEM_DATA_FILE="$STITCHED_FILE"
if [[ -d "$PERSONA_STORE_ROOT" ]]; then
    export T_MEM_PERSONA_STORE_ROOT="$PERSONA_STORE_ROOT"
fi
# Trigger configuration — default OFF (No Entity/Bridge Trigger + No Scene/Horizon, best overall hypermem result)
if [[ "$ENABLE_TRIGGERS" -eq 1 ]]; then
    # Original behaviour: keep build-stage trigger configuration
    if [[ -f "$EXP_DIR/scene_horizon_topk_per_qa.json" ]]; then
        export SCENE_HORIZON_ASSOC_TOPK_JSON="$EXP_DIR/scene_horizon_topk_per_qa.json"
    fi
else
    # Default: disable entity/bridge trigger and Scene/Horizon associative recall.
    # This aligns with the "No L1 Trigger + No L2L3" ablation config
    # which yielded the best overall score on LongMemEval (78.2±0.2).
    export T_MEM_ENTITY_BRIDGE_TRIGGER_ENABLED=0
    unset SCENE_HORIZON_ASSOC_TOPK_JSON
fi

QA_SCRIPT="$SCRIPT_DIR/qa_hypermem.py"
JUDGE_SCRIPT="$SCRIPT_DIR/judge_hypermem.py"

cd "$PROJECT_ROOT"

banner() {
    echo ""                                                             | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"                            | tee -a "$LOG_FILE"
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

banner "T_mem · LongMemEval QA+Judge (HyperMem protocol plugin)"
echo "EXP_DIR             = $EXP_DIR"             | tee -a "$LOG_FILE"
echo "LME_FILE            = $LME_FILE"            | tee -a "$LOG_FILE"
echo "STITCHED_FILE       = $STITCHED_FILE"       | tee -a "$LOG_FILE"
echo "PERSONA_STORE_ROOT  = ${T_MEM_PERSONA_STORE_ROOT:-<disabled>}" | tee -a "$LOG_FILE"
echo "QA model            = $QA_MODEL"            | tee -a "$LOG_FILE"
echo "QA concurrency      = $QA_CONCURRENCY"      | tee -a "$LOG_FILE"
echo "QA prompt           = HyperMem 7-step CoT + FINAL ANSWER extraction" | tee -a "$LOG_FILE"
echo "JUDGE_MODEL         = $JUDGE_MODEL"         | tee -a "$LOG_FILE"
echo "JUDGE_NUM_RUNS      = $NUM_RUNS"            | tee -a "$LOG_FILE"
echo "JUDGE_CONCURRENCY   = $JUDGE_CONCURRENCY"   | tee -a "$LOG_FILE"
echo "Judge protocol      = single CORRECT/WRONG + JSON label + 3-run mean±std" | tee -a "$LOG_FILE"
echo "Trigger config      = $([ "$ENABLE_TRIGGERS" -eq 1 ] && echo 'ON (build-stage)' || echo 'OFF (No Entity/Bridge Trigger + No Scene/Horizon)')" | tee -a "$LOG_FILE"

# ---- Step -1: Re-run stage6 without triggers (default behaviour) ----
# When --enable-triggers is passed, skip this — use build-stage search_results.json as-is.
if [[ "$ENABLE_TRIGGERS" -eq 0 ]]; then
    if [[ -f "$EXP_DIR/search_results.json" ]]; then
        cp "$EXP_DIR/search_results.json" "$EXP_DIR/search_results.json.bak_eval"
        trap "cp '$EXP_DIR/search_results.json.bak_eval' '$EXP_DIR/search_results.json' 2>/dev/null || true" EXIT
    fi
    run_step "Step -1 Re-run stage6 (T_MEM_ENTITY_BRIDGE_TRIGGER_ENABLED=0 + Scene/Horizon off)" \
        python3 -u -m T_mem.main.stage6_retrieval
fi

# ---- Step 0: trim stage6 wide recall (same as default) ----
if [[ "$SKIP_TRUNCATE" -eq 0 ]]; then
    run_step "Step 0 trim search_results -> final_keep_* from T_mem/config.py (in-place)" \
        python3 -u -m T_mem.io.truncate_search_results \
            --src-dir    "$EXP_DIR" \
            --dst-dir    "$EXP_DIR"
else
    banner "SKIP Step 0 truncate (--skip-truncate)"
fi

# ---- Step 1: HyperMem QA ----
HYP_FILE="$EXP_DIR/hypothesis_hypermem.jsonl"
if [[ "$SKIP_QA" -eq 0 ]]; then
    export T_MEM_QA_CONCURRENCY="$QA_CONCURRENCY"
    run_step "Step 1 HyperMem QA (7-step CoT + FINAL ANSWER extraction)" \
        python3 -u "$QA_SCRIPT" \
            --memory-dir    "$EXP_DIR" \
            --out-dir       "$EXP_DIR" \
            --stitched-file "$STITCHED_FILE" \
            --model         "$QA_MODEL" \
            --concurrency   "$QA_CONCURRENCY"
else
    banner "SKIP Step 1 QA (--skip-qa)"
fi

if [[ ! -f "$HYP_FILE" ]]; then
    echo "[eval_longmemeval_hypermem] FATAL: $HYP_FILE missing after qa_hypermem.py" >&2
    exit 3
fi

# ---- Step 2: HyperMem Judge ----
if [[ "$SKIP_JUDGE" -eq 0 ]]; then
    run_step "Step 2 HyperMem Judge (single CORRECT/WRONG + ${NUM_RUNS}-run JSON)" \
        python3 -u "$JUDGE_SCRIPT" \
            --hyp-file    "$HYP_FILE" \
            --ref-file    "$LME_FILE" \
            --out-dir     "$EXP_DIR" \
            --judge-model "$JUDGE_MODEL" \
            --num-runs    "$NUM_RUNS" \
            --concurrency "$JUDGE_CONCURRENCY"
else
    banner "SKIP Step 2 Judge (--skip-judge)"
fi

EVAL_RESULT_FILE="$EXP_DIR/$(basename "$HYP_FILE").eval-results-${JUDGE_MODEL}"
METRICS_FILE="${EVAL_RESULT_FILE}.metrics.json"

banner "ALL DONE"
echo "hypothesis_hypermem.jsonl = $HYP_FILE"                       | tee -a "$LOG_FILE"
echo "eval-results             = ${EVAL_RESULT_FILE:-<missing>}"   | tee -a "$LOG_FILE"
echo "metrics.json             = ${METRICS_FILE:-<missing>}"       | tee -a "$LOG_FILE"
echo "Log                      = $LOG_FILE"                        | tee -a "$LOG_FILE"
