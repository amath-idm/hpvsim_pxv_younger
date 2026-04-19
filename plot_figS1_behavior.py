"""
Plot sexual behavior data (Nigeria).

Two modes:
  python plot_figS1_behavior.py --run-sim   # run sim + extract CSVs (VM)
  python plot_figS1_behavior.py             # plot from saved CSVs (local)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
import seaborn as sns

import run_sims as rs
import utils as ut


def plot_sb(dist_type='lognormal', resfolder='results', outpath='figures/figS1_nigeria_behavior.png'):
    ut.set_font(14)
    fig = plt.figure(layout='tight', figsize=(12, 11))
    gs0 = fig.add_gridspec(2, 1)
    gs00 = gs0[0].subgridspec(1, 3)
    gs01 = gs0[1].subgridspec(1, 2)
    ms = 80

    # Panel A: age at first sex
    alldf = pd.read_csv(f'{resfolder}/model_sb_AFS.csv')
    sex, sk = 'Women', 'f'
    xxx = np.arange(12, 31, 1)

    ax = fig.add_subplot(gs00[0])
    for cohort in alldf['cohort'].unique():
        modely = alldf.loc[alldf['cohort'] == cohort, f'model_prop_{sk}'].values
        ax.plot(xxx, modely * 100, 'b-', lw=1, alpha=0.3)

    try:
        data_countries, dff, df2, rvs = ut.read_debut_data(dist_type=dist_type)
        dfplot = dff[sex].loc[(dff[sex]['AgeStr'] != f'{sex} never') &
                              (dff[sex]['AgeStr'] != f'{sex} 60') &
                              (dff[sex]['Country'] == 'Nigeria')]
        if len(dfplot):
            sns.scatterplot(ax=ax, data=dfplot, x='Age', y='Percentage', marker='d', s=ms, color='k')
        if 'Nigeria' in rvs[sex]:
            xx = np.arange(12, 30.1, 0.1)
            ax.plot(xx, rvs[sex]['Nigeria'].cdf(xx) * 100, 'k--', lw=2)
    except (FileNotFoundError, KeyError, IndexError):
        pass  # Nigeria DHS data not available; show model curves only

    ax.set_ylabel('Share')
    ax.set_xlabel('Age')
    ax.set_title('(A) Share of females who\n are sexually active')

    # Panel B: proportion married
    modeldf = pd.read_csv(f'{resfolder}/model_sb_prop_married.csv')
    modeldf['val'] = modeldf['val'] * 100
    modeldf['AgeRange'] = modeldf['age'].astype(str) + '-' + (modeldf['age'] + 4).astype(str)

    colors = sc.gridcolors(1)
    ax = fig.add_subplot(gs00[1])
    sns.boxplot(data=modeldf, x='AgeRange', y='val', color=colors[0], ax=ax,
                order=sorted(modeldf['AgeRange'].unique(), key=lambda s: int(s.split('-')[0])))
    try:
        dfraw = pd.read_csv('data/prop_married.csv')
        df = dfraw.melt(id_vars=['Country', 'Survey'], value_name='Percentage', var_name='AgeRange')
        df_nig = df[df['Country'] == 'Nigeria']
        if len(df_nig):
            sns.scatterplot(ax=ax, data=df_nig, x='AgeRange', y='Percentage', color='k', marker='d', s=ms)
    except FileNotFoundError:
        pass

    ax.set_ylabel('Share')
    ax.set_xlabel('Age')
    ax.set_title('(B) Share of females\n who are married')

    # Panel C: age differences between partners (precomputed KDE grid)
    kde_df = pd.read_csv(f'{resfolder}/age_diffs_kde.csv')
    ax = fig.add_subplot(gs00[2])
    ax.plot(kde_df['x'], kde_df['density'], color=colors[0])
    ax.set_xlim([-10, 30])
    ax.set_ylabel('Share')
    ax.set_xlabel('Male age - female age')
    ax.set_title('(C) Age differences\n between partners')

    # Panels D, E: degree distribution (precomputed histogram + summary stats)
    partners_hist = pd.read_csv(f'{resfolder}/partners_hist.csv')
    axlabels = ['D', 'E']
    for ai, slabel in enumerate(['females', 'males']):
        s = slabel[0]
        sub = partners_hist[partners_hist['sex'] == s].sort_values('bin')
        ax = fig.add_subplot(gs01[ai])
        ax.bar(sub['bin'].values, sub['probability'].values)
        ax.set_xlabel('Number of lifetime casual partners')
        ax.set_title(f'({axlabels[ai]}) Distribution of casual partners, {slabel}')
        row = sub.iloc[0]
        stats = (
            f'Mean: {row["mean"]:.1f}\n'
            f'Median: {row["median"]:.1f}\n'
            f'Std: {row["std"]:.1f}\n'
            f'%>20: {row["pct_gt_20"]:.2f}\n'
        )
        ax.text(15, 0.5, stats)

    fig.tight_layout()
    plt.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-sim', action='store_true',
                        help='Run the sim and save sexual-behavior CSVs (VM-side)')
    parser.add_argument('--resfolder', default='results/v2.0.x_published',
                        help='Dir with plot-ready CSVs (for plot mode only)')
    parser.add_argument('--outpath', default='figures/figS1_nigeria_behavior.png')
    args = parser.parse_args()

    if args.run_sim:
        rs.get_sb_from_sims()
        print(f'Saved sexual-behavior CSVs to results/')
    else:
        plot_sb(resfolder=args.resfolder, outpath=args.outpath)
        print('Done.')
