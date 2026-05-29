# Changelog

## v1.0.2 — 2026-05-29

Updated figures. Underlying analysis pipeline (data download, ENSO
classification, compositing) is unchanged from v1.0.1.

Added
* `scripts_v2/` — nine new figure-build scripts (`build_fig01..09.py`) plus
  a shared style module `fig_style_v2.py` (Helvetica/Arial 9 pt, cmocean
  perceptually uniform colormaps, Amazon basin and river overlays, city
  labels, bold panel labels, margin-positioned annotations and quiverkeys
  so no text overlaps data).
* `figures_v2/` — 300 DPI PNG + PDF outputs of all nine figures.
* `scripts_v2/README.md` — usage notes and data paths.

Changed
* `README.md` — describes v1 vs v2 layout; links Zenodo concept DOI.

Unchanged
* `scripts/` — original analysis pipeline preserved verbatim. This is the
  code archived in [Zenodo 10.5281/zenodo.19209348](https://doi.org/10.5281/zenodo.19209348).

## v1.0.1 — 2026-03-25

Patch release of original analysis code.

## v1.0.0 — 2026-03-24

Initial release. Code as archived in [Zenodo 10.5281/zenodo.19209348](https://doi.org/10.5281/zenodo.19209348).
