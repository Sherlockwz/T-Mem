# Project-page maintenance

GitHub Pages publishes `main:/docs`. Images, scripts and styles referenced by
the website must be inside `docs/`; `../assets/` is outside the published site.
After updating a figure under the repository's top-level `assets/`, run:

```bash
bash scripts/sync_site_assets.sh
```

Preview the same publishing root locally:

```bash
python3 -m http.server 8765 --directory docs
```

## Paper data

The September 2026 refresh uses the author-supplied camera-ready source
`T_mem_EMNLP_oveleaf/acl_latex_0526_final_submit.tex` and its `picture/` PDFs.
The Overleaf source folder is maintained separately from this repository.

- Main results: `tab:locomo`, `tab:locomoplus`.
- Model comparisons: `tab:buildmodel-qwen`, `tab:buildmodel-cross`.
- The HyperMem-matched QA protocol is a separate comparison, not a replacement
  for the official-protocol main results.
- Figures are rendered from the corresponding manuscript PDFs; no plotted
  results are estimated or redrawn from screenshots.

Update README tables and project-page tables together. Keep the author list
and BibTeX aligned with `CITATION.cff`. Until an official proceedings record is
available, retain the arXiv identifier and annotate the accepted venue.

The Pipeline Explorer reads stored examples from `demo/demo_data.json`.
Its scene artifacts are an illustrative run, not newly measured benchmark
results; do not relabel them as a new run when updating aggregate scores.
