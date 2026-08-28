#!/usr/bin/env bash
# ============================================================
# T_mem · Memory-library build script (single entry point).
#
# Two modes share this one script:
#   --mode locomo       (default)  → 10-conv main library; stages 1..7 (+4b)
#                                    used by scripts/eval_locomo.sh
#   --mode locomo_plus              → 401-conv per-sample libraries; stages 1..4 (+4b)
#                                    (each plus sample stitches its cue into
#                                    base_conv (i % 10) at last_session+7d, then
#                                    runs stages on the 401-conv stitched json).
#                                    used by scripts/eval_locomo_plus.sh
#
# Stage 4 vs 4b — two independent trigger families:
#   stage 4  Scene/Horizon triggers  (scenes/ -> scene_horizon_triggers/), read by stage 5
#            toggle: T_MEM_ENABLE_SCENE_HORIZON_TRIGGERS
#   stage 4b Entity/Bridge triggers  (memory_graphs/ -> entity_bridge_triggers/), read by stage 6
#            toggle: T_MEM_ENTITY_BRIDGE_TRIGGER_ENABLED  (governs build AND retrieval)
#
# Usage:
#   # locomo (default) – behaviour identical to the previous build_memory.sh:
#   bash scripts/build_memory.sh [--tag <name>] [--stages 1,2,3,...]
#   T_MEM_ENABLE_SCENE_HORIZON_TRIGGERS=0 bash scripts/build_memory.sh   # skip stages 4/5
#   T_MEM_ENTITY_BRIDGE_TRIGGER_ENABLED=0 bash scripts/build_memory.sh   # skip stage 4b
#
#   # locomo_plus – stitch + per-sample stages 1..4 (+4b):
#   bash scripts/build_memory.sh --mode locomo_plus [--tag <name>] [--limit N] \
#        [--stages 1,2,3,4,4b] [--locomo-plus-file <abs>]
#
# Smoke (25-sample plus) example:
#   bash scripts/build_memory.sh --mode locomo_plus --tag smoke25 --limit 25
#
# Next step:
#   bash scripts/eval_locomo.sh       --resume <experiment_dir>   # mode=locomo
#   bash scripts/eval_locomo_plus.sh  --resume <experiment_dir>   # mode=locomo_plus
#
# Model ids are declared in T_mem/config.py :: MODELS (not via shell).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------- Defaults ----------------
LOCOMO_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/locomo/data/locomo10.json"
LOCOMO_PLUS_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/locomo_plus/data/locomo_plus.json"

MODE="locomo"
TAG=""
RESUME_DIR=""
LOCOMO_FILE="${T_MEM_LOCOMO_FILE:-$LOCOMO_FILE_DEFAULT}"
LOCOMO_PLUS_FILE="${T_MEM_LOCOMO_PLUS_FILE:-$LOCOMO_PLUS_FILE_DEFAULT}"
OUT_DIR_OVERRIDE=""
STAGES=""              # filled in below per-mode after arg parse
LIMIT=""               # locomo_plus only

# ---------------- Arg parsing ----------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)             MODE="$2";             shift 2;;
        --stages)           STAGES="$2";           shift 2;;
        --tag)              TAG="$2";              shift 2;;
        --resume)           RESUME_DIR="$2";       shift 2;;
        --locomo-file)      LOCOMO_FILE="$2";      shift 2;;
        --locomo-plus-file) LOCOMO_PLUS_FILE="$2"; shift 2;;
        --out-dir)          OUT_DIR_OVERRIDE="$2"; shift 2;;
        --limit)            LIMIT="$2";            shift 2;;
        -h|--help)
            sed -n '2,37p' "$0"
            exit 0;;
        *)
            echo "[build_memory] unknown arg: $1" >&2
            exit 2;;
    esac
done

case "$MODE" in
    locomo|locomo_plus) ;;
    *) echo "[build_memory] FATAL: --mode must be locomo|locomo_plus, got: $MODE" >&2; exit 2;;
esac

# ---------------- Feature switches ----------------
# Mirrors T_mem/main/stage6_retrieval.py::_is_entity_bridge_trigger_enabled():
# default ON, only an explicit opt-out disables the feature. Keeping the two in
# sync means one variable governs BOTH the build (stage 4b) and the consumption
# (stage 6) of entity/bridge triggers.
entity_bridge_disabled() {
    local raw
    raw="$(printf '%s' "${T_MEM_ENTITY_BRIDGE_TRIGGER_ENABLED:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    case "$raw" in
        0|false|no|off) return 0;;
        *)              return 1;;
    esac
}

# ---------------- Mode-specific defaults ----------------
if [[ -z "$STAGES" ]]; then
    if [[ "$MODE" == "locomo" ]]; then
        # Stage 4b (entity/bridge triggers) is independent of the Scene/Horizon
        # switch: stage 6 consumes it, so it belongs in every locomo default.
        if [[ "${T_MEM_ENABLE_SCENE_HORIZON_TRIGGERS:-}" == "0" || \
              "${T_MEM_ENABLE_SCENE_HORIZON_TRIGGERS:-}" == "false" || \
              "${T_MEM_ENABLE_SCENE_HORIZON_TRIGGERS:-}" == "no" || \
              "${T_MEM_ENABLE_SCENE_HORIZON_TRIGGERS:-}" == "off" ]]; then
            STAGES="1,2,3,4b,6,7"
        else
            STAGES="1,2,3,4,4b,5,6,7"
        fi
    elif [[ "$MODE" == "locomo_plus" ]]; then
        # locomo_plus eval consumes only scenes/ + scene_horizon_triggers/, so stage5
        # (per-QA top-K), stage6 (retrieval), stage7 (persona) are irrelevant.
        # We still default to "1,2,3,4" -- not the leaner "1,4" -- because
        # stage 2/3 outputs (items + indexes) are cheap insurance against
        # later analysis steps that may want them. 4b is included so a plus
        # library can also be reused for entity/bridge analysis.
        STAGES="1,2,3,4,4b"
    fi
fi

# ---------------- Mode-specific input validation ----------------
if [[ "$MODE" == "locomo_plus" ]]; then
    if [[ ! -f "$LOCOMO_FILE" ]]; then
        echo "[build_memory] FATAL: $LOCOMO_FILE missing (locomo10 base file)" >&2
        exit 2
    fi
    if [[ ! -f "$LOCOMO_PLUS_FILE" ]]; then
        echo "[build_memory] FATAL: $LOCOMO_PLUS_FILE missing" >&2
        exit 2
    fi
fi

# ---------------- Experiment dir ----------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
    EXP_DIR="$OUT_DIR_OVERRIDE"
    RESULTS_ROOT="$(dirname "$EXP_DIR")"
elif [[ -n "$RESUME_DIR" ]]; then
    EXP_DIR="$RESUME_DIR"
    RESULTS_ROOT="$(dirname "$EXP_DIR")"
else
    RESULTS_ROOT="$PROJECT_ROOT/results"
    if [[ "$MODE" == "locomo_plus" ]]; then
        # Always prefix locomo_plus runs so the 401-conv per-sample libraries
        # never collide visually with the 10-conv LoCoMo main libraries.
        if [[ -n "$TAG" ]]; then
            EXP_DIR="$RESULTS_ROOT/locomo_plus_${TIMESTAMP}__${TAG}"
        else
            EXP_DIR="$RESULTS_ROOT/locomo_plus_${TIMESTAMP}"
        fi
    else
        if [[ -n "$TAG" ]]; then
            EXP_DIR="$RESULTS_ROOT/${TIMESTAMP}__${TAG}"
        else
            EXP_DIR="$RESULTS_ROOT/${TIMESTAMP}"
        fi
    fi
fi
mkdir -p "$RESULTS_ROOT" "$EXP_DIR" "$EXP_DIR/logs"

EXP_NAME="$(basename "$EXP_DIR")"
LOG_FILE="$EXP_DIR/logs/build_memory.log"
JSON_FAILURE_LOG="$EXP_DIR/logs/json_failures.jsonl"

# ---------------- Mode=locomo_plus: stage0 stitch (cue insertion) ----------------
# This produces stitched_locomo_plus.json -- a 401-conv (or N if --limit N)
# locomo10-compatible json where each plus sample's cue has been inserted
# into base_conv (i % 10) at last_session_dt + 7d. Stages 1..4 then run on
# this stitched json so each plus sample gets its own scene/trigger library.
if [[ "$MODE" == "locomo_plus" ]]; then
    DATA_DIR="$EXP_DIR/data"
    mkdir -p "$DATA_DIR"
    STITCHED_FILE="$DATA_DIR/stitched_locomo_plus.json"

    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 0 (locomo_plus stitch)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    LIMIT_ARGS=()
    if [[ -n "$LIMIT" ]]; then
        LIMIT_ARGS=(--limit "$LIMIT")
    fi
    PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
    python3 -u -m T_mem.main.stage0_locomo_plus_stitch \
        --locomo10-file    "$LOCOMO_FILE" \
        --locomo-plus-file "$LOCOMO_PLUS_FILE" \
        --out-file         "$STITCHED_FILE" \
        "${LIMIT_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        echo "[build_memory] Stage 0 (locomo_plus stitch) failed with exit $rc" | tee -a "$LOG_FILE"
        exit "$rc"
    fi
    if [[ ! -f "$STITCHED_FILE" ]]; then
        echo "[build_memory] FATAL: stitch did not produce $STITCHED_FILE" >&2
        exit 3
    fi
    # Stages 1..4 read the dataset via T_MEM_DATA_FILE; redirect to stitched.
    LOCOMO_FILE="$STITCHED_FILE"
fi

# ---------------- Env ----------------
export T_MEM_EXPERIMENT_NAME="$EXP_NAME"
export T_MEM_RESULTS_DIR="$RESULTS_ROOT"
export T_MEM_DATA_FILE="$LOCOMO_FILE"
export T_MEM_FAILURE_LOG="$JSON_FAILURE_LOG"
export T_MEM_USE_RERANKER="${T_MEM_USE_RERANKER:-true}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "============================================================"
echo "  T_mem · build_memory"
echo "============================================================"
echo "  Mode:        $MODE"
echo "  Experiment:  $EXP_DIR"
echo "  Tag:         ${TAG:-<none>}"
echo "  Dataset:     $LOCOMO_FILE"
echo "  Stages:      $STAGES"
echo "  Scene/Horizon assoc:  ${T_MEM_ENABLE_SCENE_HORIZON_TRIGGERS:-on}"
if entity_bridge_disabled; then
    echo "  Entity/Bridge assoc:  off"
else
    echo "  Entity/Bridge assoc:  on"
fi
echo "  Failure log: $JSON_FAILURE_LOG"
echo "  Model ids:   T_mem/config.py :: MODELS (single source of truth)"
echo "============================================================"

run_stage() {
    local name="$1"
    local module="$2"
    shift 2
    echo "" | tee -a "$LOG_FILE"
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] Stage $name =====" | tee -a "$LOG_FILE"
    python3 -u -m "$module" "$@" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        echo "[build_memory] Stage $name failed with exit $rc" | tee -a "$LOG_FILE"
        exit "$rc"
    fi
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] Stage $name done =====" | tee -a "$LOG_FILE"
}

PERSONA_STORE_ROOT="$EXP_DIR/personas"

IFS=',' read -ra STAGE_ARR <<< "$STAGES"
for s in "${STAGE_ARR[@]}"; do
    s="$(echo "$s" | tr -d ' ')"
    case "$s" in
        1)   run_stage "1 (memory extraction)"     "T_mem.main.stage1_memory_extraction";;
        2)   run_stage "2 (memory-graph extraction)" "T_mem.main.stage2_extraction";;
        3)   run_stage "3 (index building)"          "T_mem.main.stage3_index";;
        4)   run_stage "4 (Scene/Horizon trigger extract)" "T_mem.main.stage4_associative_extract";;
        4b)
            # Entity/Bridge triggers: built from the stage-2 memory graphs and
            # consumed by stage 6 through T_mem.retrievers.trigger_recaller.
            # Without this stage stage 6 only logs a warning and silently falls
            # back to baseline retrieval, so keep it in the default stage list.
            if entity_bridge_disabled; then
                echo "" | tee -a "$LOG_FILE"
                echo "[build_memory] Stage 4b skipped (T_MEM_ENTITY_BRIDGE_TRIGGER_ENABLED=${T_MEM_ENTITY_BRIDGE_TRIGGER_ENABLED})" | tee -a "$LOG_FILE"
            else
                if [[ ! -d "$EXP_DIR/memory_graphs" ]]; then
                    echo "[build_memory] stage 4b (Entity/Bridge trigger build) requires $EXP_DIR/memory_graphs (stage 2 output); run stage 2 first" >&2
                    exit 2
                fi
                run_stage "4b (Entity/Bridge trigger build)" "T_mem.main.build_trigger" \
                    --memory-dir "$EXP_DIR" \
                    --output-dir "$EXP_DIR/entity_bridge_triggers"
            fi
            ;;
        5)
            # stage5_retrieval_locomo is LoCoMo-specific. LoCoMo-Plus builds its
            # per-sample top-K inside scripts/eval_locomo_plus.sh (which runs
            # stage5_retrieval_locomo_plus), so refuse instead of silently
            # running the wrong module.
            if [[ "$MODE" != "locomo" ]]; then
                echo "[build_memory] stage 5 is only valid for --mode locomo (got: $MODE)." >&2
                echo "[build_memory] For LoCoMo-Plus run: bash scripts/eval_locomo_plus.sh --resume $EXP_DIR" >&2
                exit 2
            fi
            run_stage "5 (per-QA top-K build)"     "T_mem.main.stage5_retrieval_locomo"
            if [[ -f "$EXP_DIR/scene_horizon_topk_per_qa.json" ]]; then
                export SCENE_HORIZON_ASSOC_TOPK_JSON="$EXP_DIR/scene_horizon_topk_per_qa.json"
                echo "[build_memory] SCENE_HORIZON_ASSOC_TOPK_JSON=$SCENE_HORIZON_ASSOC_TOPK_JSON" | tee -a "$LOG_FILE"
            fi
            ;;
        6)   run_stage "6 (retrieval)"              "T_mem.main.stage6_retrieval";;
        7)
            if [[ ! -d "$EXP_DIR/scenes" ]]; then
                echo "[build_memory] stage 7 (persona build) requires $EXP_DIR/scenes (stage 1 output); run stage 1 first" >&2
                exit 2
            fi
            mkdir -p "$PERSONA_STORE_ROOT"
            echo "" | tee -a "$LOG_FILE"
            echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] Stage 7 (persona build) =====" | tee -a "$LOG_FILE"
            python3 -u -m T_mem.main.stage7_persona \
                --scenes-root "$EXP_DIR/scenes" \
                --store-root    "$PERSONA_STORE_ROOT" \
                --concurrency   "${T_MEM_PERSONA_CONCURRENCY:-1}" 2>&1 | tee -a "$LOG_FILE"
            rc=${PIPESTATUS[0]}
            if [[ "$rc" -ne 0 ]]; then
                echo "[build_memory] Stage 7 (persona build) failed with exit $rc" | tee -a "$LOG_FILE"
                exit "$rc"
            fi
            echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] Stage 7 (persona build) done =====" | tee -a "$LOG_FILE"
            ;;
        *)   echo "[build_memory] unknown stage: $s" >&2; exit 2;;
    esac
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  Memory library build complete"        | tee -a "$LOG_FILE"
echo "  Mode:         $MODE"                  | tee -a "$LOG_FILE"
echo "  Experiment:   $EXP_DIR"               | tee -a "$LOG_FILE"
echo "  Log:          $LOG_FILE"              | tee -a "$LOG_FILE"
echo ""                                       | tee -a "$LOG_FILE"
echo "  Next:"                                | tee -a "$LOG_FILE"
if [[ "$MODE" == "locomo" ]]; then
    echo "    bash scripts/eval_locomo.sh      --resume $EXP_DIR" | tee -a "$LOG_FILE"
else
    echo "    bash scripts/eval_locomo_plus.sh --resume $EXP_DIR" | tee -a "$LOG_FILE"
fi
echo "============================================================" | tee -a "$LOG_FILE"
