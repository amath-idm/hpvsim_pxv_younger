"""
Plot calibration to Nigeria.

Two modes:
  python plot_figS2_calibration.py --run-sim   # run calibration + extract CSVs (VM)
  python plot_figS2_calibration.py             # plot from saved CSVs (local)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
import seaborn as sns

import run_sims as rs
import utils as ut


def save_figS2_data(calib, res_to_plot=100, resfolder='results'):
    n = min(res_to_plot, len(calib.analyzer_results))

    per_trial = np.array([calib.analyzer_results[pos]['cancers'][2020] for pos in range(n)])
    pd.DataFrame({
        'bin': np.arange(per_trial.shape[1]),
        'median': np.median(per_trial, axis=0),
        'pi95_low': np.percentile(per_trial, 2.5, axis=0),
        'pi95_high': np.percentile(per_trial, 97.5, axis=0),
    }).to_csv(f'{resfolder}/figS2_cancers_by_age.csv', index=False)

    for rkey in ['cin_genotype_dist', 'cancerous_genotype_dist']:
        per_trial = []
        for pos in range(n):
            run = calib.sim_results[pos][rkey]
            per_trial.append([run] if sc.isnumber(run) else list(run))
        per_trial = np.array(per_trial)
        rows = []
        for bi in range(per_trial.shape[1]):
            arr = per_trial[:, bi]
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            iqr = q3 - q1
            lo = float(arr[arr >= q1 - 1.5 * iqr].min())
            hi = float(arr[arr <= q3 + 1.5 * iqr].max())
            rows.append({'bin': bi, 'q1': float(q1), 'med': float(med), 'q3': float(q3),
                         'whislo': lo, 'whishi': hi})
        pd.DataFrame(rows).to_csv(f'{resfolder}/figS2_{rkey}.csv', index=False)

    for ti, name in enumerate(['cancers', 'cin_genotype', 'cancerous_genotype']):
        if ti < len(calib.target_data):
            calib.target_data[ti].to_csv(f'{resfolder}/figS2_target_{name}.csv', index=False)


def plot_calib(resfolder='results', outpath='figures/figS2_calibration.png'):
    ut.set_font(size=24)
    fig = plt.figure(layout='tight', figsize=(12, 11))
    canc_col = '#c1981d'
    ms = 80
    gen_cols = sc.gridcolors(4)

    gs0 = fig.add_gridspec(2, 1)
    gs00 = gs0[0].subgridspec(1, 1)
    gs01 = gs0[1].subgridspec(1, 2)

    # Panel A: Cancers by age, 2020
    cancers_df = pd.read_csv(f'{resfolder}/figS2_cancers_by_age.csv')
    target_cancers = pd.read_csv(f'{resfolder}/figS2_target_cancers.csv')
    age_labels = ['0', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85']
    x = np.arange(len(age_labels))

    ax = fig.add_subplot(gs00[0])
    cancers_df = cancers_df.sort_values('bin')
    ax.plot(cancers_df['bin'], cancers_df['median'], color=canc_col)
    ax.fill_between(cancers_df['bin'], cancers_df['pi95_low'],
                    cancers_df['pi95_high'], color=canc_col, alpha=0.3)
    ax.scatter(x, target_cancers['value'].values, marker='d', s=ms, color='k')
    ax.set_ylim([0, 2_500])
    ax.set_xticks(x, age_labels)
    sc.SIticks(ax)
    ax.set_xlabel('Age')
    ax.set_title('Cancers by age, 2020')

    # Panels B, C: CIN + cancer by genotype
    for ai, (rkey, target_name, rlabel) in enumerate([
        ('cin_genotype_dist', 'cin_genotype', 'HSILs'),
        ('cancerous_genotype_dist', 'cancerous_genotype', 'Cancers'),
    ]):
        ax = fig.add_subplot(gs01[ai])
        model_df = pd.read_csv(f'{resfolder}/figS2_{rkey}.csv')
        target_df = pd.read_csv(f'{resfolder}/figS2_target_{target_name}.csv')

        model_df = model_df.sort_values('bin')
        bxp_stats = [dict(med=r.med, q1=r.q1, q3=r.q3,
                          whislo=r.whislo, whishi=r.whishi, fliers=[],
                          label=str(int(r.bin))) for _, r in model_df.iterrows()]
        bp = ax.bxp(bxp_stats, positions=model_df['bin'].values,
                    patch_artist=True, showfliers=False, manage_ticks=False)
        for patch, color in zip(bp['boxes'], gen_cols):
            patch.set_facecolor(color)
        ax.scatter(np.arange(len(target_df)), target_df['value'].values, color='k', marker='d', s=ms)
        ax.set_ylim([0, 1])
        ax.set_xticks(np.arange(4), ['16', '18', 'Hi5', 'OHR'])
        ax.set_title(rlabel)

    fig.tight_layout()
    plt.savefig(outpath, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-sim', action='store_true',
                        help='Run calibration and save CSVs (heavy, VM-side)')
    parser.add_argument('--resfolder', default='results/v2.0.x_published',
                        help='Dir with plot-ready CSVs (for plot mode only)')
    parser.add_argument('--outpath', default='figures/figS2_calibration.png')
    parser.add_argument('--res-to-plot', type=int, default=100)
    args = parser.parse_args()

    if args.run_sim:
        sim, calib = rs.run_calib(n_trials=rs.n_trials, n_workers=rs.n_workers,
                                  do_save=True, filestem='')
        save_figS2_data(calib, res_to_plot=args.res_to_plot, resfolder='results')
        print('Saved figS2 CSVs to results/ (copy to a versioned baseline dir to commit)')
    else:
        plot_calib(resfolder=args.resfolder, outpath=args.outpath)
        print('Done.')
