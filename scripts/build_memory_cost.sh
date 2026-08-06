#!/usr/bin/env bash
# ============================================================
# T_mem · FULL LoCoMo construction build WITH per-LLM-call cost logging.
#
# Reproduces the complete write-time pipeline (so the resulting library is
# QA-ready) while logging every construction LLM call to a JSONL ledger.
# The single build serves BOTH:
#   * the construction-cost experiment (R1-W1), and
#   * the memory library + search_results.json that the cross-judge (R1-W2)
#     and stronger-backbone (R2-W1) experiments consume.
#
# Stage order (item-level Entity+Bridge triggers `trig` run BEFORE stage6,
# because stage6 retrieval loads them from <exp>/entity_bridge_triggers/):
#   1 scene -> 2 graph -> 3 index -> 4 scene-trigger -> entity_bridge(item-trigger)
#     -> 5 per-QA topk -> 6 retrieval -> 7 persona
#
# Only additions vs a normal build:
#   T_MEM_COST_LOG / T_MEM_COST_STAGE  (cost ledger)
#   T_MEM_MEMORY_BUILD_MODEL           (default gpt-4.1-mini == paper)
#
# Usage:
#   bash scripts/build_memory_cost.sh --tag <name> [--model <id>] \
#        [--out-dir <abs>] [--stages "1,2,3,4,trig,5,6,7"]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LOCOMO_FILE="${T_MEM_LOCOMO_FILE:-$PROJECT_ROOT/benchmark_eval/locomo/data/locomo10.json}"
MODEL="gpt-4.1-mini"
TAG="locomo_cost"
STAGES="1,2,3,4,trig,5,6,7"
OUT_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)         TAG="$2";     shift 2;;
        --model)       MODEL="$2";   shift 2;;
        --out-dir)     OUT_DIR_OVERRIDE="$2"; shift 2;;
        --stages)      STAGES="$2";  shift 2;;
        --locomo-file) LOCOMO_FILE="$2"; shift 2;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0;;
        *) echo "[build_memory_cost] unknown arg: $1" >&2; exit 2;;
    esac
done

# ---------------- Preflight ----------------
if [[ ! -f "$LOCOMO_FILE" ]]; then
    echo "[build_memory_cost] FATAL: dataset missing: $LOCOMO_FILE" >&2; exit 2
fi
if [[ -f "$PROJECT_ROOT/.env" ]]; then set -a; . "$PROJECT_ROOT/.env"; set +a; fi
if [[ -z "${ENV_VENUS_OPENAPI_SECRET_ID:-}" ]]; then
    echo "[build_memory_cost] FATAL: ENV_VENUS_OPENAPI_SECRET_ID not set. Aborting." >&2; exit 2
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
    EXP_DIR="$OUT_DIR_OVERRIDE"; RESULTS_ROOT="$(dirname "$EXP_DIR")"
else
    RESULTS_ROOT="$PROJECT_ROOT/results"; EXP_DIR="$RESULTS_ROOT/cost_${TIMESTAMP}__${TAG}"
fi
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/cost" "$RESULTS_ROOT"

EXP_NAME="$(basename "$EXP_DIR")"
LOG_FILE="$EXP_DIR/logs/build.log"
COST_LOG="$EXP_DIR/cost/llm_calls.jsonl"

export T_MEM_EXPERIMENT_NAME="$EXP_NAME"
export T_MEM_RESULTS_DIR="$RESULTS_ROOT"
export T_MEM_DATA_FILE="$LOCOMO_FILE"
export T_MEM_FAILURE_LOG="$EXP_DIR/logs/json_failures.jsonl"
export T_MEM_USE_RERANKER="${T_MEM_USE_RERANKER:-true}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export T_MEM_MEMORY_BUILD_MODEL="$MODEL"
export T_MEM_COST_LOG="$COST_LOG"

echo "============================================================" | tee -a "$LOG_FILE"
echo "  build_memory_cost | exp=$EXP_DIR" | tee -a "$LOG_FILE"
echo "  build model=$MODEL | stages=$STAGES | cost_log=$COST_LOG"    | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

run_stage() {  # <label> <cost_stage> <module> [args...]
    local label="$1"; local cost_stage="$2"; local module="$3"; shift 3
    echo "" | tee -a "$LOG_FILE"
    echo "===== [$(date '+%F %T')] $label (cost_stage=$cost_stage) =====" | tee -a "$LOG_FILE"
    T_MEM_COST_STAGE="$cost_stage" python3 -u -m "$module" "$@" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then echo "[build_memory_cost] $label failed (exit=$rc)" | tee -a "$LOG_FILE"; exit "$rc"; fi
    echo "===== [$(date '+%F %T')] $label done =====" | tee -a "$LOG_FILE"
}

IFS=',' read -ra STAGE_ARR <<< "$STAGES"
for s in "${STAGE_ARR[@]}"; do
    s="$(echo "$s" | tr -d ' ')"
    case "$s" in
        1) run_stage "Stage 1 (scene)"           "stage1" "T_mem.main.stage1_memory_extraction";;
        2) run_stage "Stage 2 (memory graph)"    "stage2" "T_mem.main.stage2_extraction";;
        3) run_stage "Stage 3 (index/embed)"     "stage3" "T_mem.main.stage3_index";;
        4) run_stage "Stage 4 (scene triggers)"  "stage4" "T_mem.main.stage4_associative_extract";;
        trig)
            run_stage "Item-level triggers (Entity+Bridge)" "build_trigger" \
                "T_mem.main.build_trigger" --memory-dir "$EXP_DIR" --output-dir "$EXP_DIR/trigger";;
        5)
            run_stage "Stage 5 (per-QA topk)"    "stage5" "T_mem.main.stage5_retrieval_locomo"
if [[ -f "$EXP_DIR/scene_horizon_topk_per_qa.json" ]]; then
    export SCENE_HORIZON_ASSOC_TOPK_JSON="$EXP_DIR/scene_horizon_topk_per_qa.json"
    echo "[build_memory_cost] SCENE_HORIZON_ASSOC_TOPK_JSON=$SCENE_HORIZON_ASSOC_TOPK_JSON" | tee -a "$LOG_FILE"
            fi;;
        6) run_stage "Stage 6 (retrieval)"       "stage6" "T_mem.main.stage6_retrieval";;
        7)
            mkdir -p "$EXP_DIR/personas"
            run_stage "Stage 7 (persona)" "stage7" "T_mem.main.stage7_persona" \
                --scenes-root "$EXP_DIR/scenes" --store-root "$EXP_DIR/personas" \
                --concurrency "${T_MEM_PERSONA_CONCURRENCY:-1}";;
        *) echo "[build_memory_cost] unknown stage token: $s" >&2; exit 2;;
    esac
done

echo "" | tee -a "$LOG_FILE"
echo "[build_memory_cost] BUILD DONE: $EXP_DIR" | tee -a "$LOG_FILE"
echo "  cost ledger: $COST_LOG" | tee -a "$LOG_FILE"
echo "  search_results.json present: $([[ -f "$EXP_DIR/search_results.json" ]] && echo yes || echo NO)" | tee -a "$LOG_FILE"
# Emit the resolved experiment dir on the LAST line for callers to capture.
echo "EXP_DIR=$EXP_DIR"
