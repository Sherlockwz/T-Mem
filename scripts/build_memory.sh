#!/usr/bin/env bash
# ============================================================
# T_mem · Memory-library build script (single entry point).
#
# Three modes share this one script:
#   --mode locomo       (default)  → 10-conv main library; stages 1..7
#                                    used by scripts/eval_locomo.sh
#   --mode locomo_plus              → 401-conv per-sample libraries; stages 1..4
#                                    (each plus sample stitches its cue into
#                                    base_conv (i % 10) at last_session+7d, then
#                                    runs stages on the 401-conv stitched json).
#                                    used by scripts/eval_locomo_plus.sh
#   --mode lme                      → 500-conv per-instance libraries; stages 1..7
#                                    (LongMemEval-S; each instance is its own
#                                    haystack history of 38–62 sessions, stitched
#                                    into a locomo10-shape conversation by
#                                    stage0_lme_stitch). Full T-Mem capability
#                                    enabled (scenes + items + index + L2L3 +
#                                    main retrieval + persona).
#                                    used by scripts/eval_longmemeval.sh
#
# Usage:
#   # locomo (default) – behaviour identical to the previous build_memory.sh:
#   bash scripts/build_memory.sh [--tag <name>] [--stages 1,2,3,...]
#   T_MEM_ENABLE_L2L3=0 bash scripts/build_memory.sh   # skip stages 4/5
#
#   # locomo_plus – stitch + per-sample stages 1..4:
#   bash scripts/build_memory.sh --mode locomo_plus [--tag <name>] [--limit N] \
#        [--stages 1,2,3,4] [--locomo-plus-file <abs>]
#
#   # lme – stitch LongMemEval-S + per-instance stages 1..7:
#   bash scripts/build_memory.sh --mode lme [--tag <name>] [--limit N] \
#        [--stages 1,2,3,4,5,6,7] [--lme-file <abs>]
#
# Smoke (25-sample plus) example:
#   bash scripts/build_memory.sh --mode locomo_plus --tag smoke25 --limit 25
#
# Smoke (14-instance lme) example:
#   bash scripts/build_memory.sh --mode lme --tag smoke14 \
#        --lme-file benchmark_eval/longmemeval/data/longmemeval_s_smoke14.json
#
# Next step:
#   bash scripts/eval_locomo.sh       --resume <experiment_dir>   # mode=locomo
#   bash scripts/eval_locomo_plus.sh  --resume <experiment_dir>   # mode=locomo_plus
#   bash scripts/eval_longmemeval.sh  --resume <experiment_dir>   # mode=lme
#
# Model ids are declared in T_mem/config.py :: MODELS (not via shell).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------- Defaults ----------------
LOCOMO_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/locomo/data/locomo10.json"
LOCOMO_PLUS_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/locomo_plus/data/locomo_plus.json"
LME_FILE_DEFAULT="$PROJECT_ROOT/benchmark_eval/longmemeval/data/longmemeval_s_cleaned.json"

MODE="locomo"
TAG=""
RESUME_DIR=""
LOCOMO_FILE="${T_MEM_LOCOMO_FILE:-$LOCOMO_FILE_DEFAULT}"
LOCOMO_PLUS_FILE="${T_MEM_LOCOMO_PLUS_FILE:-$LOCOMO_PLUS_FILE_DEFAULT}"
LME_FILE="${T_MEM_LME_FILE:-$LME_FILE_DEFAULT}"
OUT_DIR_OVERRIDE=""
STAGES=""              # filled in below per-mode after arg parse
LIMIT=""               # locomo_plus / lme only

# ---------------- Arg parsing ----------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)             MODE="$2";             shift 2;;
        --stages)           STAGES="$2";           shift 2;;
        --tag)              TAG="$2";              shift 2;;
        --resume)           RESUME_DIR="$2";       shift 2;;
        --locomo-file)      LOCOMO_FILE="$2";      shift 2;;
        --locomo-plus-file) LOCOMO_PLUS_FILE="$2"; shift 2;;
        --lme-file)         LME_FILE="$2";         shift 2;;
        --out-dir)          OUT_DIR_OVERRIDE="$2"; shift 2;;
        --limit)            LIMIT="$2";            shift 2;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0;;
        *)
            echo "[build_memory] unknown arg: $1" >&2
            exit 2;;
    esac
done

case "$MODE" in
    locomo|locomo_plus|lme) ;;
    *) echo "[build_memory] FATAL: --mode must be locomo|locomo_plus|lme, got: $MODE" >&2; exit 2;;
esac

# ---------------- Mode-specific defaults ----------------
if [[ -z "$STAGES" ]]; then
    if [[ "$MODE" == "locomo" ]]; then
        # Mirror the original (pre-merge) build_memory.sh defaults exactly so
        # `--mode locomo` keeps byte-for-byte legacy behaviour.
        if [[ "${T_MEM_ENABLE_L2L3:-}" == "0" || \
              "${T_MEM_ENABLE_L2L3:-}" == "false" || \
              "${T_MEM_ENABLE_L2L3:-}" == "no" || \
              "${T_MEM_ENABLE_L2L3:-}" == "off" ]]; then
            STAGES="1,2,3,6,7"
        else
            STAGES="1,2,3,4,5,6,7"
        fi
    elif [[ "$MODE" == "locomo_plus" ]]; then
        # locomo_plus eval consumes only scenes/ + l2l3_triggers/, so stage5
        # (per-QA top-K), stage6 (retrieval), stage7 (persona) are irrelevant.
        # We still default to "1,2,3,4" -- not the leaner "1,4" -- because
        # stage 2/3 outputs (items + indexes) are cheap insurance against
        # later analysis steps that may want them, and this matches the
        # build_memory_locomo_plus.sh defaults that produced the 77.31 result.
        STAGES="1,2,3,4"
    else
        # lme: full T-Mem capability per the user requirement ("all memory
        # libraries should be present"). Mirrors the LoCoMo full-stage default.
        STAGES="1,2,3,4,5,6,7"
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

if [[ "$MODE" == "lme" ]]; then
    if [[ ! -f "$LME_FILE" ]]; then
        echo "[build_memory] FATAL: $LME_FILE missing (LongMemEval data file)" >&2
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
    elif [[ "$MODE" == "lme" ]]; then
        # Always prefix lme runs so the 500-conv per-instance libraries are
        # visually distinct from LoCoMo / LoCoMo-Plus runs in $RESULTS_ROOT.
        if [[ -n "$TAG" ]]; then
            EXP_DIR="$RESULTS_ROOT/lme_${TIMESTAMP}__${TAG}"
        else
            EXP_DIR="$RESULTS_ROOT/lme_${TIMESTAMP}"
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

# ---------------- Mode=lme: stage0 stitch (LongMemEval -> locomo10-shape) ----------------
# Produces stitched_lme.json -- one locomo10-shape conversation per LongMemEval
# question instance (38–62 sessions / instance, 500 instances total). Stages
# 1..7 then run on this stitched json so each instance gets its own scene /
# trigger / persona library.
if [[ "$MODE" == "lme" ]]; then
    DATA_DIR="$EXP_DIR/data"
    mkdir -p "$DATA_DIR"
    STITCHED_FILE="$DATA_DIR/stitched_lme.json"

    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 0 (lme stitch)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    LIMIT_ARGS=()
    if [[ -n "$LIMIT" ]]; then
        LIMIT_ARGS=(--limit "$LIMIT")
    fi
    PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
    python3 -u -m T_mem.main.stage0_lme_stitch \
        --lme-file  "$LME_FILE" \
        --out-file  "$STITCHED_FILE" \
        "${LIMIT_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        echo "[build_memory] Stage 0 (lme stitch) failed with exit $rc" | tee -a "$LOG_FILE"
        exit "$rc"
    fi
    if [[ ! -f "$STITCHED_FILE" ]]; then
        echo "[build_memory] FATAL: stitch did not produce $STITCHED_FILE" >&2
        exit 3
    fi
    # Stages 1..7 read the dataset via T_MEM_DATA_FILE; redirect to stitched.
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
echo "  L2L3 assoc:  ${T_MEM_ENABLE_L2L3:-on}"
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
        4)   run_stage "4 (L2/L3 trigger extract)" "T_mem.main.stage4_associative_extract";;
        5)
            run_stage "5 (per-QA top-K build)"     "T_mem.main.stage5_retrieval_locomo"
            if [[ -f "$EXP_DIR/l2l3_topk_per_qa.json" ]]; then
                export L2L3_ASSOC_TOPK_JSON="$EXP_DIR/l2l3_topk_per_qa.json"
                echo "[build_memory] L2L3_ASSOC_TOPK_JSON=$L2L3_ASSOC_TOPK_JSON" | tee -a "$LOG_FILE"
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
elif [[ "$MODE" == "locomo_plus" ]]; then
    echo "    bash scripts/eval_locomo_plus.sh --resume $EXP_DIR" | tee -a "$LOG_FILE"
else
    echo "    bash scripts/eval_longmemeval.sh --resume $EXP_DIR" | tee -a "$LOG_FILE"
fi
echo "============================================================" | tee -a "$LOG_FILE"
