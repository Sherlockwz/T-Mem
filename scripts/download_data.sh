#!/usr/bin/env bash
# ============================================================
# T-Mem · Dataset download script
#
# Downloads the three public benchmarks used in the paper into
# benchmark_eval/<benchmark>/data/. All datasets are publicly
# available; no credentials required.
#
#   LoCoMo       -> benchmark_eval/locomo/data/locomo10.json
#   LoCoMo-Plus  -> benchmark_eval/locomo_plus/data/locomo_plus.json
#   LongMemEval  -> benchmark_eval/longmemeval/data/longmemeval_s_cleaned.json
#
# Usage:
#   bash scripts/download_data.sh            # download everything
#   bash scripts/download_data.sh locomo     # just one benchmark
#   bash scripts/download_data.sh locomo_plus
#   bash scripts/download_data.sh longmemeval
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"

_dl() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "  [skip] $dest already exists"
        return 0
    fi
    echo "  [download] $url"
    mkdir -p "$(dirname "$dest")"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$dest" "$url"
    else
        echo "  [error] neither wget nor curl found" >&2
        return 1
    fi
}

locomo() {
    local dest="$PROJECT_ROOT/benchmark_eval/locomo/data/locomo10.json"
    echo "[locomo]"
    _dl "https://raw.githubusercontent.com/snap-research/LoCoMo/main/data/locomo10.json" "$dest"
}

locomo_plus() {
    local dest="$PROJECT_ROOT/benchmark_eval/locomo_plus/data/locomo_plus.json"
    echo "[locomo_plus]"
    _dl "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo_plus.json" "$dest"
}

longmemeval() {
    local dest="$PROJECT_ROOT/benchmark_eval/longmemeval/data/longmemeval_s_cleaned.json"
    echo "[longmemeval]"
    _dl "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" "$dest"
}

case "$TARGET" in
    all)
        locomo
        locomo_plus
        longmemeval
        ;;
    locomo)
        locomo
        ;;
    locomo_plus)
        locomo_plus
        ;;
    longmemeval)
        longmemeval
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh [all|locomo|locomo_plus|longmemeval]" >&2
        exit 1
        ;;
esac

echo "Done."
