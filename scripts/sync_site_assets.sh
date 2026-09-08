#!/usr/bin/env bash
# GitHub Pages publishes docs/ only. Keep its figures in that publishing root.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/docs/assets"
for figure in figure1 figure2 figure3_ablation figure4_hyperparam figure5_token_accuracy; do
  cp "$repo_dir/assets/$figure.png" "$repo_dir/docs/assets/$figure.png"
done
