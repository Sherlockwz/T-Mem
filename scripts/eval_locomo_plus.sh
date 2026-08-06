#!/usr/bin/env bash
# ============================================================
# T_mem · LoCoMo-Plus QA + judge (requires an existing memory library).
#
# Shares scenes + scene_horizon_triggers with LoCoMo, but uses a LoCoMo-Plus specific
# retrieval path:
#   Step 1  stage5_retrieval_locomo_plus  → locomo_plus_topk_per_sample.json
#              (3-channel (dialogue / scene / horizon) cosine → RRF fusion, top-K per sample)
#   Step 2  stage8_qa_locomo_plus    → predictions.json
#   Step 3  run_judge.py             → judge.json + metrics.json
#                                       (reads predictions.json from EXP_DIR
#                                       directly via --predictions; no mount step)
#
# Usage:
#   bash scripts/eval_locomo_plus.sh --resume <experiment_dir>
#   bash scripts/eval_locomo_plus.sh --resume <experiment_dir> --topk 5 --content summary
#
# Model ids are declared in T_mem/config.py :: MODELS.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LOCOMO_PLUS_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/locomo_plus/data/locomo_plus.json"
LOCOMO10_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/locomo/data/locomo10.json"

EXP_DIR=""
LOCOMO_PLUS_FILE="${T_MEM_LOCOMO_PLUS_FILE:-$LOCOMO_PLUS_FILE_DEFAULT}"
LOCOMO10_FILE="${T_MEM_LOCOMO10_FILE:-$LOCOMO10_FILE_DEFAULT}"
TOPK="${T_MEM_LOCOMO_PLUS_TOPK:-10}"
CONTENT="${T_MEM_LOCOMO_PLUS_CONTENT:-summary}"   # summary | description | dialogue
JUDGE_PROMPT="${T_MEM_LOCOMO_PLUS_JUDGE_PROMPT:-A_paper_original}"
JUDGE_NUM_RUNS="${T_MEM_JUDGE_NUM_RUNS:-1}"
JUDGE_CONCURRENCY="${T_MEM_JUDGE_CONCURRENCY:-16}"
SKIP_STAGE5=0
SKIP_STAGE8=0
SKIP_JUDGE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)             EXP_DIR="$2";             shift 2;;
        --locomo-plus-file)   LOCOMO_PLUS_FILE="$2";    shift 2;;
        --locomo10-file)      LOCOMO10_FILE="$2";       shift 2;;
        --topk)               TOPK="$2";                shift 2;;
        --content)            CONTENT="$2";             shift 2;;
        --judge-prompt)       JUDGE_PROMPT="$2";        shift 2;;
        --judge-num-runs)     JUDGE_NUM_RUNS="$2";      shift 2;;
        --judge-concurrency)  JUDGE_CONCURRENCY="$2";   shift 2;;
        --skip-stage5)        SKIP_STAGE5=1;            shift;;
        --skip-stage8)        SKIP_STAGE8=1;            shift;;
        --skip-judge)         SKIP_JUDGE=1;             shift;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0;;
        *)
            echo "[eval_locomo_plus] unknown arg: $1" >&2
            exit 2;;
    esac
done

# ---------------- Preflight ----------------
if [[ -z "$EXP_DIR" ]]; then
    echo "[eval_locomo_plus] FATAL: --resume <experiment_dir> is required" >&2
    echo "  Run scripts/build_memory.sh first to produce the memory library." >&2
    exit 2
fi
EXP_DIR="$(cd "$EXP_DIR" && pwd)"
if [[ ! -d "$EXP_DIR/scenes" ]]; then
    echo "[eval_locomo_plus] FATAL: $EXP_DIR/scenes missing (stage 1 not done?)" >&2
    exit 2
fi
if [[ ! -d "$EXP_DIR/scene_horizon_triggers" ]]; then
    echo "[eval_locomo_plus] FATAL: $EXP_DIR/scene_horizon_triggers missing (stage 4 not done?)" >&2
    echo "  LoCoMo-Plus retrieval requires Scene/Horizon triggers; stage 4 is mandatory." >&2
    exit 2
fi
if [[ ! -f "$LOCOMO_PLUS_FILE" ]]; then
    echo "[eval_locomo_plus] FATAL: $LOCOMO_PLUS_FILE missing" >&2
    exit 2
fi
# locomo10.json is now optional: per-sample mode reads speakers from the
# stitched dataset. Only fail if neither stitched nor locomo10 is available.
STITCHED_FILE="$EXP_DIR/data/stitched_locomo_plus.json"
if [[ ! -f "$STITCHED_FILE" && ! -f "$LOCOMO10_FILE" ]]; then
    echo "[eval_locomo_plus] FATAL: neither $STITCHED_FILE nor $LOCOMO10_FILE found (need at least one as speaker source)" >&2
    exit 2
fi

LOG_DIR="$EXP_DIR/logs"
LOG_FILE="$LOG_DIR/eval_locomo_plus.log"
mkdir -p "$LOG_DIR"

# Method-tag used as the judge output subdir name only. (No mount step; the
# judge reads predictions.json from EXP_DIR directly.)
EXP_NAME="$(basename "$EXP_DIR")"
METHOD_TAG="t_mem__${EXP_NAME}__top${TOPK}_${CONTENT}"
JUDGE_RESULTS_ROOT="$EXP_DIR/locomo_plus_judge"

# ---------------- Env ----------------
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export T_MEM_EXPERIMENT_NAME="$EXP_NAME"
export T_MEM_RESULTS_DIR="$(dirname "$EXP_DIR")"
# LoCoMo-Plus does not use LoCoMo's dataset_path, but stage5_retrieval_locomo_plus imports
# stage5_retrieval_locomo which reads ExperimentConfig paths. We pass the
# locomo_plus file via argparse instead of env.
export T_MEM_LOCOMO_PLUS_FILE="$LOCOMO_PLUS_FILE"
export T_MEM_LOCOMO10_FILE="$LOCOMO10_FILE"
if [[ -f "$STITCHED_FILE" ]]; then
    export T_MEM_STITCHED_PLUS_FILE="$STITCHED_FILE"
fi
export T_MEM_FAILURE_LOG="${T_MEM_FAILURE_LOG:-$LOG_DIR/json_failures.jsonl}"
export T_MEM_DATA_FILE="${T_MEM_DATA_FILE:-$LOCOMO10_FILE}"

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

banner "T_mem · LoCoMo-Plus QA+Judge"
echo "EXP_DIR             = $EXP_DIR"             | tee -a "$LOG_FILE"
echo "LOCOMO_PLUS_FILE    = $LOCOMO_PLUS_FILE"    | tee -a "$LOG_FILE"
echo "LOCOMO10_FILE       = $LOCOMO10_FILE"       | tee -a "$LOG_FILE"
if [[ -f "$STITCHED_FILE" ]]; then
    echo "STITCHED_FILE       = $STITCHED_FILE  [per-sample mode]" | tee -a "$LOG_FILE"
else
    echo "STITCHED_FILE       = <not found>     [legacy idx %% n_conv mode]" | tee -a "$LOG_FILE"
fi
echo "TOPK                = $TOPK"                | tee -a "$LOG_FILE"
echo "CONTENT             = $CONTENT"             | tee -a "$LOG_FILE"
echo "METHOD_TAG          = $METHOD_TAG"          | tee -a "$LOG_FILE"
echo "JUDGE_PROMPT        = $JUDGE_PROMPT"        | tee -a "$LOG_FILE"
echo "Model ids           = T_mem/config.py :: MODELS" | tee -a "$LOG_FILE"

# ---- Step 1/3: stage5_retrieval_locomo_plus (retrieval) ----
if [[ "$SKIP_STAGE5" -eq 0 ]]; then
    run_step "Step 1/3 stage5_retrieval_locomo_plus (per-sample top-K via 3-way RRF)" \
        python3 -u -m T_mem.main.stage5_retrieval_locomo_plus
else
    banner "SKIP Step 1/3 stage5_retrieval_locomo_plus (--skip-stage5)"
fi

TOPK_FILE="$EXP_DIR/locomo_plus_topk_per_sample.json"
if [[ ! -f "$TOPK_FILE" ]]; then
    echo "[eval_locomo_plus] FATAL: $TOPK_FILE missing after stage5_retrieval_locomo_plus" >&2
    exit 3
fi

# ---- Step 2/3: stage8_qa_locomo_plus (QA) ----
if [[ "$SKIP_STAGE8" -eq 0 ]]; then
    STAGE8_SPEAKER_ARGS=()
    if [[ -f "$STITCHED_FILE" ]]; then
        STAGE8_SPEAKER_ARGS+=(--stitched-file "$STITCHED_FILE")
    fi
    if [[ -f "$LOCOMO10_FILE" ]]; then
        STAGE8_SPEAKER_ARGS+=(--locomo10-file "$LOCOMO10_FILE")
    fi
    run_step "Step 2/3 stage8_qa_locomo_plus (per-sample QA)" \
        python3 -u -m T_mem.main.stage8_qa_locomo_plus \
            --memory-dir "$EXP_DIR" \
            --out-dir    "$EXP_DIR" \
            --topk       "$TOPK" \
            --content    "$CONTENT" \
            --locomo-plus-file "$LOCOMO_PLUS_FILE" \
            "${STAGE8_SPEAKER_ARGS[@]}"
else
    banner "SKIP Step 2/3 stage8_qa_locomo_plus (--skip-stage8)"
fi

PRED_SRC="$EXP_DIR/predictions.json"
if [[ ! -f "$PRED_SRC" ]]; then
    echo "[eval_locomo_plus] FATAL: $PRED_SRC missing after stage8_qa_locomo_plus" >&2
    exit 3
fi

# ---- Step 3/3: run_judge.py (Locomo-Plus LLM-as-judge) ----
# run_judge.py reads predictions.json directly via --predictions; no mount step.
if [[ "$SKIP_JUDGE" -eq 0 ]]; then
    run_step "Step 3/3 run_judge.py (Locomo-Plus)" \
        python3 -u -m benchmark_eval.locomo_plus.judge.run_judge \
            --predictions   "$PRED_SRC" \
            --method-tag    "$METHOD_TAG" \
            --prompt        "$JUDGE_PROMPT" \
            --num-runs      "$JUDGE_NUM_RUNS" \
            --concurrency   "$JUDGE_CONCURRENCY" \
            --results-root  "$JUDGE_RESULTS_ROOT"
else
    banner "SKIP Step 3/3 run_judge.py (--skip-judge)"
fi

banner "ALL DONE"
echo "topk cache          = $TOPK_FILE"                       | tee -a "$LOG_FILE"
echo "predictions.json    = $PRED_SRC"                        | tee -a "$LOG_FILE"
echo "judge results root  = $JUDGE_RESULTS_ROOT"              | tee -a "$LOG_FILE"
echo "Log                 = $LOG_FILE"                        | tee -a "$LOG_FILE"
