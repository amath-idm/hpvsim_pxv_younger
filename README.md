# Prophylactic HPV vaccination for infants (Nigeria)

Code for analysing the impact of moving the HPV prophylactic vaccine to the infant series.

**Results in this repository were produced with HPVsim v2.2.6.** Plot-ready baseline CSVs from the original (v2.0.x-era, Nov 2024) analysis live in [`results/v2.0.x_published/`](results/v2.0.x_published/); working CSVs for the current version live at the top of [`results/`](results/).

## Installation

```bash
pip install hpvsim==2.2.6 seaborn optuna
```

Python 3.9+.

## Workflow: heavy sims on VM, plots locally

Each plot script has two modes: `--run-sim` runs the heavy calculation and saves lightweight CSVs in `results/`; running without flags loads the CSVs and produces the figure. The intended flow is:

1. **VM:** `python plot_XXX.py --run-sim` → produces CSVs
2. Commit & push from VM
3. **Local:** `python plot_XXX.py` → renders the figure from CSVs

Large binaries (`vs.msim`, full calibration objects, raw per-event CSVs) are gitignored — only plot-ready CSVs are committed.

## Running scripts

| Script | Figure | Heavy step | Key artifacts |
|---|---|---|---|
| `plot_fig2_bars.py` | Fig 2 — cancers/deaths averted by efficacy and coverage | `run_scenarios.py` with `efficacy_scen='all'` | `fig23_scens_all.csv` |
| `plot_fig3_ts.py` | Fig 3 — time series comparing equivalent efficacy scenarios | `run_scenarios.py` with `efficacy_scen='equiv'` | `fig23_scens_equiv.csv` |
| `plot_figS1_behavior.py` | Fig S1 — sexual behavior | `get_sb_from_sims()` + `run_degree.py` | `model_sb_AFS.csv`, `model_sb_prop_married.csv`, `model_age_diffs.csv`, `model_casual.csv`, `partners.csv` |
| `plot_figS2_calibration.py` | Fig S2 — calibration | Full calibration (needs many trials) | `figS2_cancers_by_age.csv`, `figS2_cin_genotype_dist.csv`, `figS2_cancerous_genotype_dist.csv` + 3 target CSVs |
| `plot_figS3_age_pyramids.py` | Fig S3 — age pyramids over time | Baseline sim with `age_pyramid` analyzer | `figS3_model.csv`, `figS3_data.csv` |
| `plot_fig_lines.py` | Analytical efficacy/coverage lines | — | self-contained |

## Heavy-step scripts

- `run_sims.py`: sim-building, calibration, sexual-behavior extraction (`get_sb_from_sims`)
- `run_scenarios.py`: full vaccination-scenario msim run; set `efficacy_scen = 'all'` or `'equiv'` at the top. Saves both `.obj` and `fig23_scens_*.csv`.
- `run_degree.py`: casual-partner degree distribution extraction (saves `partners.csv`)

## Baseline utilities

- `save_baselines.py`: one-shot extractor that re-generates plot-ready CSVs from a set of source `.obj` files (e.g. the v2.0.x snapshot).

## Adding a future-version baseline (v2.3, v3.0, ...)

When a new HPVsim version ships, produce a frozen baseline for comparison:

```bash
# 1. On VM, in a clean environment pinned to the new version
conda create -n hpvsim230 python=3.11 -y && conda activate hpvsim230
pip install hpvsim==2.3.0 seaborn optuna
# 2. Regenerate the CSVs (each script mode is heavy)
python run_scenarios.py              # set efficacy_scen='all', then 'equiv', re-run
python plot_figS1_behavior.py --run-sim
python plot_figS2_calibration.py --run-sim
python plot_figS3_age_pyramids.py --run-sim
python run_degree.py
# 3. Freeze into a versioned baseline dir (copy, don't move — keep working copy in results/)
mkdir -p results/v2.3.0_baseline
cp results/*.csv results/v2.3.0_baseline/
# 4. Commit and push the baseline alongside an updated manifest.json.
```

Compare across versions by plotting from a baseline dir via `--resfolder`, e.g.
`python plot_figS1_behavior.py --resfolder results/v2.0.x_published` vs. the
default `results/` (current version). For side-by-side views, write a small
comparison script that takes `--baselines` (see the `compare_fig2.py` pattern
in [hpvsim_india](https://github.com/hpvsim/hpvsim_india)).

## Inputs

- `data/` — input data files (Nigeria cancer cases, cancer types, CIN types, HPV prevalence, age pyramid, ASR cancer, plus shared DHS files `afs_dist.csv`, `afs_median.csv`, `prop_married.csv` copied from the India repo).
- `nigeria_age_pyramid.csv` at repo root — population pyramid data.

## Further information

See [hpvsim.org](https://hpvsim.org) and [docs.hpvsim.org](https://docs.hpvsim.org).
