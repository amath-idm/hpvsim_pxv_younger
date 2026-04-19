# Changelog

## 2026-04-18 — HPVsim v2.2.6 lift

- Split every plot script into a VM-side `--run-sim` step that saves lightweight CSVs and a local plot step that reads them.
- Added `save_baselines.py` to (re)generate plot-ready CSVs from a set of source `.obj` files.
- Froze plot-ready baselines from the Nov 2024 (v2.0.x-era) analysis into `results/v2.0.x_published/`; working CSVs for the current version live at the top of `results/`.
- Updated `run_sims.py`, `run_scenarios.py`, `run_degree.py` to additionally emit CSVs alongside `.obj` on VM runs.
- Fixed figS1 Panel B: boxplots and DHS scatter now share the AgeRange axis.
- Dropped MySQL `storage` requirement from calibration.
- Added `.gitignore` excluding large binaries (`vs.msim`, full calibration objects, raw per-event CSVs).
- Added MIT LICENSE.
- Refreshed README with the new workflow and pinned HPVsim version.

## 2024-11 — v2.0.x era

- Original Nigeria calibration, scenarios, and supplementary analyses for the infant HPV vaccination paper.
