'''
Define an HPVsim simulation for Nigeria
'''

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv
import pandas as pd

# Imports from this repository
import utils as ut

# %% Settings and filepaths

# Debug switch
debug = 0  # Run with smaller population sizes and in serial 
do_shrink = True  # Do not keep people when running sims (saves memory)

# Run settings
n_trials    = [4000, 2][debug]  # How many trials to run for calibration
n_workers   = [50, 1][debug]    # How many cores to use
# storage     = ["mysql://hpvsim_user@localhost/hpvsim_newdb", None][debug]  # Storage for calibrations
storage = None

# Save settings
do_save = True
save_plots = True


# %% Simulation creation functions
def make_sim(location='nigeria', calib_pars=None, debug=0, interventions=None, analyzers=None, datafile=None, seed=1, end=2020):
    """
    Define parameters, analyzers, and interventions for the simulation
    """

    # Basic parameters
    pars = sc.objdict(
        n_agents=[20e3, 1e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 1980][debug],
        end=end,
        genotypes=[16, 18, 'hi5', 'ohr'],
        location=location,
        ms_agent_ratio=100,
        verbose=0.0,
        rand_seed=seed,
    )

    # Sexual behavior parameters
    # Debut: derived by fitting to 2018 DHS
    # Women:
    #           Age:   15,   18,   20,   22,   25
    #   Prop_active: 19.2, 57.3, 73.9, 85.5, 91.6
    # Men:
    #           Age:  15,   18,   20,   22,   25
    #   Prop_active: 3.1, 14.5, 30.1, 51.9, 70.1
    # For fitting, see https://www.researchsquare.com/article/rs-3074559/v1
    pars.debut = dict(
        f=dict(dist='lognormal', par1=16., par2=4),
        m=dict(dist='lognormal', par1=18., par2=4),
    )

    # Participation in marital and casual relationships
    # Derived to fit 2018 DHS data
    # For fitting, see https://www.researchsquare.com/article/rs-3074559/v1
    pars.layer_probs = dict(
        m=np.array([
            # Share of people of each age who are married
            [0, 5, 10,    15,     20,     25,     30,     35,     40,     45,   50,   55,   60,   65,    70,   75],
            # [0, 0,  0,  0.1596, 0.4466, 0.5845, 0.6139, 0.6202, 0.6139, 0.5726, 0.35, 0.21, 0.14, 0.07, 0.035, 0.007],
            [0, 0,  0,  0.1,     0.1,    0.15,    0.15,    0.15,   0.2,    0.3,  0.4,  0.4,  0.2, 0.07, 0.035, 0.007],
            [0, 0,  0,  0.1,     0.1,    0.15,    0.15,    0.2,    0.2,    0.4,  0.4,  0.4,  0.2,  0.1,  0.05, 0.01 ],
        ]),
        c=np.array([
            # Share of people of each age in casual partnerships
            [0, 5,  10,  15,  20,  25,  30,   35,   40,   45,  50,  55,   60,   65,   70,   75],
            [0,  0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4,  0.7,  0.7, 0.6, 0.2, 0.10, 0.02, 0.02, 0.02],
            [0,  0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4,  0.5,  0.6, 0.5, 0.2, 0.02, 0.02, 0.02, 0.02]
        ])
    )

    pars.m_partners = dict(
        m=dict(dist='poisson1', par1=0.01),
        c=dict(dist='poisson1', par1=0.2),
    )
    pars.f_partners = dict(
        m=dict(dist='poisson1', par1=0.01),
        c=dict(dist='poisson1', par1=0.2),
    )

    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    if analyzers is None:
        analyzers = []

    sim = hpv.Sim(pars=pars, interventions=interventions, analyzers=analyzers, datafile=datafile)

    return sim


# %% Simulation running functions
def run_sim(calib_pars=None, analyzers=None, debug=debug, datafile=None, seed=1, verbose=.1, do_shrink=do_shrink, do_save=do_save, end=2020):
    # Make sim
    sim = make_sim(
        debug=debug,
        seed=seed,
        datafile=datafile,
        analyzers=analyzers,
        calib_pars=calib_pars,
        end=end
    )
    sim.label = f'Sim-{seed}'

    # Run
    sim['verbose'] = verbose
    sim.run()
    if do_shrink:
        sim.shrink()

    # Optionally save
    if do_save:
        sim.save(f'results/nigeria.sim')

    return sim


def run_calib(n_trials=None, n_workers=None, do_save=True, filestem=''):

    sim = make_sim()
    datafiles = [
        f'data/nigeria_cancer_cases.csv',
        f'data/nigeria_cin_types.csv',
        f'data/nigeria_cancer_types.csv',
    ]

    # Define the calibration parameters
    genotype_pars = dict(
        hi5=dict(
            cancer_fn=dict(transform_prob=[1.5e-3, 0.5e-3, 2.5e-3, 2e-4]),
            cin_fn=dict(k=[.15, .1, .25, 0.01]),
            dur_cin=dict(par1=[4.5, 3.5, 5.5, 0.5], par2=[20, 16, 24, 0.5]),
        ),
        ohr=dict(
            cancer_fn=dict(transform_prob=[1.5e-3, 0.5e-3, 2.5e-3, 2e-4]),
            cin_fn=dict(k=[.15, .1, .25, 0.01]),
            dur_cin=dict(par1=[4.5, 3.5, 5.5, 0.5], par2=[20, 16, 24, 0.5]),
        ),
    )

    calib_pars = dict(
        beta=[0.2, 0.1, 0.34, 0.02],
        m_cross_layer=[0.3, 0.1, 0.7, 0.05],
        m_partners=dict(
            c=dict(par1=[0.2, 0.1, 0.6, 0.02])
        ),
        f_cross_layer=[0.1, 0.05, 0.5, 0.05],
        f_partners=dict(
            c=dict(par1=[0.2, 0.1, 0.6, 0.02])
        ),
        sev_dist=dict(par1=[1, 0.5, 1.5, 0.01])
    )

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            name=f'nigeria_calib',
                            datafiles=datafiles,
                            total_trials=n_trials, n_workers=n_workers,
                            storage=storage
                            )
    calib.calibrate()
    filename = f'nigeria_calib{filestem}'
    if do_save:
        sc.saveobj(f'results/{filename}.obj', calib)

    print(f'Best pars are {calib.best_pars}')

    return sim, calib


def get_sb_from_sims(verbose=-1, calib_pars=None, debug=False):
    '''
    Run sims with the sexual debut parameters inferred from DHS data, and save
    the proportion of people of each age who've ever had sex
    '''


    sim = run_sim(
        calib_pars=calib_pars,
        analyzers=[ut.AFS(), ut.prop_married(), hpv.snapshot(timepoints=['2020'])],
        debug=debug,
        verbose=verbose,
    )

    # Save output on age at first sex (AFS)
    dfs = sc.autolist()
    a = sim.get_analyzer('AFS')
    for cs, cohort_start in enumerate(a.cohort_starts):
        df = pd.DataFrame()
        df['age'] = a.bins
        df['cohort'] = cohort_start
        df['model_prop_f'] = a.prop_active_f[cs, :]
        df['model_prop_m'] = a.prop_active_m[cs, :]
        dfs += df
    afs_df = pd.concat(dfs)
    sc.saveobj(f'results/model_sb_AFS.obj', afs_df)

    # Save output on proportion married
    a = sim.get_analyzer('prop_married')
    pm_df = a.df
    sc.saveobj(f'results/model_sb_prop_married.obj', pm_df)

    # Save output on age differences between partners
    agediff_df = pd.DataFrame()
    snapshot = sim.get_analyzer('snapshot')
    ppl = snapshot.snapshots[0]
    age_diffs = ppl.contacts['m']['age_m'] - ppl.contacts['m']['age_f']
    agediff_df['age_diffs'] = age_diffs
    sc.saveobj(f'results/model_age_diffs.obj', agediff_df)

    # Save output on the number of casual relationships
    binspan = 5
    bins = np.arange(15, 50, binspan)
    snapshot = sim.get_analyzer('snapshot')
    ppl = snapshot.snapshots[0]
    conditions = {}
    general_conditions = ppl.is_female * ppl.alive * ppl.level0 * ppl.is_active
    for ab in bins:
        conditions[ab] = (ppl.age >= ab) * (ppl.age < ab + binspan) * general_conditions

    casual_partners = {(0, 1): sc.autolist(), (1, 2): sc.autolist(), (2, 3): sc.autolist(),
                       (3, 5): sc.autolist(), (5, 50): sc.autolist()}
    for cp in casual_partners.keys():
        for ab, age_cond in conditions.items():
            this_condition = conditions[ab] * (ppl.current_partners[1, :] >= cp[0]) * (
                    ppl.current_partners[1, :] < cp[1])
            casual_partners[cp] += len(hpv.true(this_condition))

    popsize = sc.autolist()
    for ab, age_cond in conditions.items():
        popsize += len(hpv.true(age_cond))

    # Construct dataframe
    n_bins = len(bins)
    partners = np.repeat([0, 1, 2, 3, 5], n_bins)
    allbins = np.tile(bins, 5)
    counts = np.concatenate([val for val in casual_partners.values()])
    allpopsize = np.tile(popsize, 5)
    shares = counts / allpopsize
    datadict = dict(bins=allbins, partners=partners, counts=counts, popsize=allpopsize, shares=shares)
    casual_df = pd.DataFrame.from_dict(datadict)

    sc.saveobj(f'results/model_casual.obj', casual_df)

    return sim, afs_df, pm_df, agediff_df, casual_df


def plot_calib(which_pars=0, save_pars=True, filestem=''):
    filename = f'nigeria_calib{filestem}'
    calib = sc.load(f'results/{filename}.obj')

    sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
    sc.options(font='Libertinus Sans')
    fig = calib.plot(res_to_plot=200, plot_type='sns.boxplot', do_save=False)
    fig.tight_layout()
    fig.savefig(f'figures/{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        trial_pars = sc.autolist()
        for i in range(100):
            trial_pars += calib.trial_pars_to_sim_pars(which_pars=i)
        sc.save(f'results/nigeria_pars{filestem}.obj', calib_pars)
        sc.save(f'results/nigeria_pars{filestem}_all.obj', trial_pars)

    return calib


def run_parsets(debug=False, verbose=.1, analyzers=None, save_results=True, **kwargs):
    ''' Run multiple simulations in parallel '''

    parsets = sc.loadobj(f'results/nigeria_pars_all.obj')
    kwargs = sc.mergedicts(dict(debug=debug, end=2040, verbose=verbose, analyzers=analyzers), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(calib_pars=parsets), kwargs=kwargs, serial=debug, die=True)
    msim = hpv.MultiSim(simlist)
    msim.reduce()
    if save_results:
        sc.saveobj(f'results/nigeria_msim.obj', msim.results)

    return msim


# %% Run as a script
if __name__ == '__main__':

    # List of what to run
    to_run = [
        # 'run_sim',
        # 'get_behavior',
        'age_pyramids',
        # 'run_calib',
        # 'plot_calib'
        # 'run_parsets'
    ]

    T = sc.timer()  # Start a timer

    if 'run_sim' in to_run:
        calib_pars = sc.loadobj('results/nigeria_pars.obj')  # Load parameters from a previous calibration
        sim = run_sim(calib_pars=calib_pars, do_save=False, do_shrink=True)  # Run the simulation
        sim.plot()  # Plot the simulation

    if 'get_behavior' in to_run:
        calib_pars = sc.loadobj('results/nigeria_pars.obj')
        # calib_pars = None
        sim, afs_df, pm_df, agediff_df, casual_df = get_sb_from_sims(calib_pars=calib_pars)

    if 'age_pyramids' in to_run:
        calib_pars = sc.loadobj('results/nigeria_pars.obj')
        ap = hpv.age_pyramid(
            timepoints=['2025', '2050', '2075', '2100'],
            datafile='nigeria_age_pyramid.csv',
            edges=np.linspace(0, 100, 21),
        )
        sim = run_sim(end=2100, calib_pars=calib_pars, analyzers=[ap], do_save=True, do_shrink=True)

    if 'run_calib' in to_run:
        sim, calib = run_calib(n_trials=n_trials, n_workers=n_workers, filestem='', do_save=True)

    if 'plot_calib' in to_run:
        calib = plot_calib(save_pars=True, filestem='')
        calib = ut.shrink_calib(calib, n_results=200)
        sc.saveobj(f'results/nigeria_calib_reduced.obj', calib)

    if 'run_parsets' in to_run:
        msim = run_parsets()

    T.toc('Done')  # Print out how long the run took
