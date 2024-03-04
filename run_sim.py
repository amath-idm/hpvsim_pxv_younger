"""
Define the HPVsim simulation
"""

# Additions to handle numpy multithreading
import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import analyzers as an

# %% Settings and filepaths
# Debug switch
debug = 0  # Run with smaller population sizes and in serial
do_shrink = True  # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


# %% Simulation creation functions
def make_sim(calib_pars=None, debug=0, interventions=None, datafile=None, seed=1, end=2020):
    """
    Define parameters, analyzers, and interventions for the simulation
    """

    # Basic parameters
    pars = sc.objdict(
        n_agents=[10e3, 1e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 1980][debug],
        end=end,
        genotypes=[16, 18, 'hi5', 'ohr'],
        location='nigeria',
        init_hpv_dist=dict(hpv16=0.4, hpv18=0.25, hi5=0.25, ohr=.1),
        init_hpv_prev={
            'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
            'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio=100,
        verbose=0.0,
        rand_seed=seed,
    )

    # Sexual behavior parameters
    pars.debut = dict(
        f=dict(dist='lognormal', par1=17.41, par2=2.75),
        m=dict(dist='lognormal', par1=17.91, par2=2.83),
    )
    pars.layer_probs = dict(
        m=np.array([
            # Share of people of each age who are married
            [0, 5,  10,       15,     20,     25,     30,     35,     40,     45,   50,   55,   60,   65,   70,   75],
            [0, 0, 0.05,  0.1596, 0.4466, 0.5845, 0.6139, 0.6202, 0.6139, 0.5726, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],  # Females
            [0, 0, 0.01,    0.01,   0.10,   0.50,   0.60,   0.70,   0.70,   0.70, 0.70, 0.80, 0.70, 0.60, 0.50, 0.60]]  # Males
        ),
        c=np.array([
            # Share of people of each age in casual partnerships
            [0, 5,  10,  15,  20,  25,  30,  35,  40,   45,   50,   55,   60,   65,   70,   75],
            [0, 0, 0.1, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0, 0, 0.1, 0.3, 0.4, 0.3, 0.3, 0.4, 0.5, 0.50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]
        ),
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

    analyzers = [an.econ_analyzer()]

    # Interventions
    sim = hpv.Sim(pars=pars, interventions=interventions, analyzers=analyzers, datafile=datafile)

    return sim


# %% Simulation running functions
def run_sim(
        interventions=None, debug=0, seed=1, verbose=0.2,
        do_save=False, end=2020, calib_pars=None, meta=None):

    dflocation = location.replace(' ', '_')

    # Make sim
    sim = make_sim(
        debug=debug,
        interventions=interventions,
        calib_pars=calib_pars,
        end=end,
    )
    sim['rand_seed'] = seed
    sim.label = f'{location}--{seed}'

    # Store metadata
    sim.meta = sc.objdict()
    if meta is not None:
        sim.meta = meta  # Copy over meta info
    else:
        sim.meta = sc.objdict()
    sim.meta.location = location  # Store location in an easy-to-access place

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()

    if do_save:
        sim.save(f'results/{dflocation}.sim')

    return sim


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    location = 'nigeria'
    calib_par_stem = '_nov13'
    calib_pars = sc.loadobj(f'results/{location}_pars{calib_par_stem}.obj')
    sim = run_sim(calib_pars=calib_pars, end=2020)
    sim.plot()

    T.toc('Done')

