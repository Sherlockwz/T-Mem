#!/usr/bin/env bash
# ============================================================
# T_mem · LongMemEval QA + judge (requires an existing memory library).
#
# Consumes build_memory.sh --mode lme outputs (search_results.json + personas
# + stitched_lme.json) and runs:
#   Step 0  T_mem.io.truncate_search_results → trims stage6 wide recall
#                                     (24 scene + 40 item by default) down to
#                                     final_keep_scene / final_keep_item
#                                     (default 5 / 15 == paper-final M1_N15
#                                     cell). Override via env:
#                                     T_MEM_FINAL_KEEP_SCENE / T_MEM_FINAL_KEEP_ITEM.
#   Step 1  stage8_qa_lme         → responses.json + hypothesis.jsonl
#                                   (hypothesis.jsonl == LongMemEval official
#                                    schema fed straight into the judge).
#   Step 2  benchmark_eval/longmemeval/judge/run_judge.py
#                                 → hypothesis.jsonl.eval-results-<judge>
#                                 + hypothesis.jsonl.eval-results-<judge>.metrics.json
#                                   (Overall / Task-averaged / Abstention
#                                    accuracy + 6-qtype breakdown.)
#
# Usage:
#   bash scripts/eval_longmemeval.sh --resume <experiment_dir>
#   bash scripts/eval_longmemeval.sh --resume <experiment_dir> --judge-model gpt-4o
#   bash scripts/eval_longmemeval.sh --skip-truncate --resume <experiment_dir>
#
# Reader prompt + reader model are 100 % aligned with LongMemEval official:
#   reader prompt = ANSWER_PROMPT_LME_COT (== upstream `reading_method=con`)
#   reader model  = MODELS["locomo_qa"]   (== gpt-4o-mini, upstream baseline)
#   judge prompt  = byte-for-byte clone of evaluate_qa.py (5+1 templates)
#   judge model   = gpt-4o                (== upstream default)
#
# Model ids are declared in T_mem/config.py :: MODELS (reader) and via
# --judge-model on this script (judge).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LME_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/longmemeval/data/longmemeval_s_cleaned.json"

EXP_DIR=""
LME_FILE="${T_MEM_LME_FILE:-$LME_FILE_DEFAULT}"
ANSWER_PROMPT="${T_MEM_LME_ANSWER_PROMPT:-lme_cot}"   # lme_cot (reading_method=con) | lme (reading_method=direct)
JUDGE_MODEL="${T_MEM_LME_JUDGE_MODEL:-gpt-4o}"
JUDGE_CONCURRENCY="${T_MEM_LME_JUDGE_CONCURRENCY:-16}"
QA_CONCURRENCY="${T_MEM_QA_CONCURRENCY:-14}"
QA_MODEL_OVERRIDE=""
SKIP_TRUNCATE=0
SKIP_QA=0
SKIP_JUDGE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)             EXP_DIR="$2";             shift 2;;
        --lme-file)           LME_FILE="$2";            shift 2;;
        --answer-prompt)      ANSWER_PROMPT="$2";       shift 2;;
        --qa-model)           QA_MODEL_OVERRIDE="$2";   shift 2;;
        --qa-concurrency)     QA_CONCURRENCY="$2";      shift 2;;
        --judge-model)        JUDGE_MODEL="$2";         shift 2;;
        --judge-concurrency)  JUDGE_CONCURRENCY="$2";   shift 2;;
        --skip-truncate)      SKIP_TRUNCATE=1;          shift;;
        --skip-qa)            SKIP_QA=1;                shift;;
        --skip-judge)         SKIP_JUDGE=1;             shift;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0;;
        *)
            echo "[eval_longmemeval] unknown arg: $1" >&2
            exit 2;;
    esac
done

# ---------------- Preflight ----------------
if [[ -z "$EXP_DIR" ]]; then
    echo "[eval_longmemeval] FATAL: --resume <experiment_dir> is required" >&2
    echo "  Run scripts/build_memory.sh --mode lme first to produce the memory library." >&2
    exit 2
fi
EXP_DIR="$(cd "$EXP_DIR" && pwd)"

if [[ ! -f "$EXP_DIR/search_results.json" ]]; then
    echo "[eval_longmemeval] FATAL: $EXP_DIR/search_results.json missing (stage 6 not done?)" >&2
    exit 2
fi

STITCHED_FILE="$EXP_DIR/data/stitched_lme.json"
if [[ ! -f "$STITCHED_FILE" ]]; then
    echo "[eval_longmemeval] FATAL: $STITCHED_FILE missing (stage 0 lme stitch not done?)" >&2
    exit 2
fi

if [[ ! -f "$LME_FILE" ]]; then
    echo "[eval_longmemeval] FATAL: $LME_FILE missing (LongMemEval reference data)" >&2
    exit 2
fi

PERSONA_STORE_ROOT="$EXP_DIR/personas"
LOG_DIR="$EXP_DIR/logs"
LOG_FILE="$LOG_DIR/eval_longmemeval.log"
mkdir -p "$LOG_DIR"

# ---------------- Env ----------------
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export T_MEM_FAILURE_LOG="${T_MEM_FAILURE_LOG:-$LOG_DIR/json_failures.jsonl}"
# Experiment env consistent with build_memory.sh so any stage8 / stage5 imports
# that read ExperimentConfig keep pointing at the right experiment dir.
export T_MEM_EXPERIMENT_NAME="$(basename "$EXP_DIR")"
export T_MEM_RESULTS_DIR="$(dirname "$EXP_DIR")"
export T_MEM_DATA_FILE="$STITCHED_FILE"          # stage6 / truncate read this
if [[ -d "$PERSONA_STORE_ROOT" ]]; then
    export T_MEM_PERSONA_STORE_ROOT="$PERSONA_STORE_ROOT"
fi
# Pick up Scene/Horizon per-QA top-K if stage 5 produced it; stage6 / stage8_qa_lme
# downstream tooling honours this env var the same way as eval_locomo.sh.
if [[ -f "$EXP_DIR/scene_horizon_topk_per_qa.json" ]]; then
    export SCENE_HORIZON_ASSOC_TOPK_JSON="$EXP_DIR/scene_horizon_topk_per_qa.json"
fi

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

banner "T_mem · LongMemEval QA+Judge"
echo "EXP_DIR             = $EXP_DIR"             | tee -a "$LOG_FILE"
echo "LME_FILE            = $LME_FILE"            | tee -a "$LOG_FILE"
echo "STITCHED_FILE       = $STITCHED_FILE"       | tee -a "$LOG_FILE"
echo "PERSONA_STORE_ROOT  = ${T_MEM_PERSONA_STORE_ROOT:-<disabled>}" | tee -a "$LOG_FILE"
echo "ANSWER_PROMPT       = $ANSWER_PROMPT"       | tee -a "$LOG_FILE"
echo "QA_CONCURRENCY      = $QA_CONCURRENCY"      | tee -a "$LOG_FILE"
echo "JUDGE_MODEL         = $JUDGE_MODEL"         | tee -a "$LOG_FILE"
echo "JUDGE_CONCURRENCY   = $JUDGE_CONCURRENCY"   | tee -a "$LOG_FILE"
echo "Reader model id     = T_mem/config.py :: MODELS['locomo_qa'] (gpt-4o-mini)" | tee -a "$LOG_FILE"

# ---- Step 0/2: trim stage6 wide recall to final_keep_scene / final_keep_item ----
# Same logic as eval_locomo.sh; the trim K is read from
# T_mem/config.py :: ExperimentConfig.retrieval_config (default 5/15).
if [[ "$SKIP_TRUNCATE" -eq 0 ]]; then
    run_step "Step 0/2 trim search_results -> final_keep_* from T_mem/config.py (in-place)" \
        python3 -u -m T_mem.io.truncate_search_results \
            --src-dir    "$EXP_DIR" \
            --dst-dir    "$EXP_DIR"
else
    banner "SKIP Step 0/2 truncate (--skip-truncate)"
fi

# ---- Step 1/2: stage8_qa_lme (per-instance QA + hypothesis.jsonl) ----
HYP_FILE="$EXP_DIR/hypothesis.jsonl"
if [[ "$SKIP_QA" -eq 0 ]]; then
    QA_MODEL_ARGS=()
    if [[ -n "$QA_MODEL_OVERRIDE" ]]; then
        QA_MODEL_ARGS+=(--model "$QA_MODEL_OVERRIDE")
    fi
    export T_MEM_QA_CONCURRENCY="$QA_CONCURRENCY"
    run_step "Step 1/2 stage8_qa_lme (per-instance QA over stage6 retrieval)" \
        python3 -u -m T_mem.main.stage8_qa_lme \
            --memory-dir    "$EXP_DIR" \
            --out-dir       "$EXP_DIR" \
            --stitched-file "$STITCHED_FILE" \
            --answer-prompt "$ANSWER_PROMPT" \
            --concurrency   "$QA_CONCURRENCY" \
            "${QA_MODEL_ARGS[@]}"
else
    banner "SKIP Step 1/2 stage8_qa_lme (--skip-qa)"
fi

if [[ ! -f "$HYP_FILE" ]]; then
    echo "[eval_longmemeval] FATAL: $HYP_FILE missing after stage8_qa_lme" >&2
    exit 3
fi

# ---- Step 2/2: run_judge.py (LongMemEval official LLM-as-judge) ----
if [[ "$SKIP_JUDGE" -eq 0 ]]; then
    run_step "Step 2/2 run_judge.py (LongMemEval official judge)" \
        python3 -u -m benchmark_eval.longmemeval.judge.run_judge \
            --hyp-file    "$HYP_FILE" \
            --ref-file    "$LME_FILE" \
            --out-dir     "$EXP_DIR" \
            --judge-model "$JUDGE_MODEL" \
            --concurrency "$JUDGE_CONCURRENCY"
else
    banner "SKIP Step 2/2 run_judge.py (--skip-judge)"
fi

EVAL_RESULT_FILE="$EXP_DIR/$(basename "$HYP_FILE").eval-results-${JUDGE_MODEL}"
METRICS_FILE="${EVAL_RESULT_FILE}.metrics.json"

banner "ALL DONE"
echo "hypothesis.jsonl    = $HYP_FILE"                          | tee -a "$LOG_FILE"
echo "eval-results        = ${EVAL_RESULT_FILE:-<missing>}"      | tee -a "$LOG_FILE"
echo "metrics.json        = ${METRICS_FILE:-<missing>}"          | tee -a "$LOG_FILE"
echo "Log                 = $LOG_FILE"                           | tee -a "$LOG_FILE"
