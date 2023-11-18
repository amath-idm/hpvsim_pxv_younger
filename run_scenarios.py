'''
Run HPVsim scenarios for each location. 

Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
'''


#%% General settings

import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp
import analyzers as an

# Comment out to not run
to_run = [
    'run_scenarios',
    # 'plot_scenarios'
]

# Comment out locations to not run
locations = [
    # 'india',        # 0
    # 'indonesia',    # 1
    'nigeria',      # 2
    # 'tanzania',     # 3
    # 'bangladesh',   # 4
    # 'myanmar',      # 5
    # 'uganda',       # 6
    # 'ethiopia',     # 7
    # 'drc',          # 8
    # 'kenya'         # 9
]

debug = 0
n_seeds = [3, 1][debug] # How many seeds to run per cluster



#%% Functions

def make_msims(sims, use_mean=True):
    '''
    Utility to take a slice of sims and turn it into a multisim
    '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_sc, i_vx, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except parameter set matches
        assert i_sc == sim.meta.inds[0]
        assert i_vx == sim.meta.inds[1]
        assert (s == 0) or i_s != sim.meta.inds[2]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_sc, i_vx]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')

    print(f'Processing multisim {msim.meta.vals.values()}...')

    return msim

def run_scens(location=None, screen_intvs=None, vx_intvs=None, # Input data
              debug=0, n_seeds=n_seeds, verbose=-1,# Sim settings
              calib_filestem=''
              ):
    '''
    Run all screening/triage product scenarios for a given location
    '''

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(screen_intvs) * len(vx_intvs) * n_seeds

    for i_sc, sc_label, screen_scen_pars in screen_intvs.enumitems():
        for i_vx, vx_label, vx_scen_pars in vx_intvs.enumitems():
            for i_s in range(n_seeds):  # n samples per cluster
                count += 1
                meta = sc.objdict()
                meta.count = count
                meta.n_sims = n_sims
                meta.inds = [i_sc, i_vx, i_s]
                meta.vals = sc.objdict(sc.mergedicts(screen_scen_pars, vx_scen_pars,
                                                     dict(seed=i_s, screen_scen=sc_label,
                                                          vx_scen=vx_label)))
                ikw.append(sc.objdict(screen_intv=screen_scen_pars, vx_intv=vx_scen_pars,
                                      seed=i_s))
                ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    calib_pars = sc.loadobj(f'results/{location}_pars{calib_filestem}.obj')
    kwargs = dict(calib_pars=calib_pars, verbose=verbose, debug=debug, location=location,
                  econ_analyzer=True, end=2100, n_agents=50e3)
    n_workers = 40
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs, ncpus=n_workers)

    # Rearrange sims
    sims = np.empty((len(screen_intvs), len(vx_intvs), n_seeds), dtype=object)
    econdfs = sc.autolist()
    for sim in all_sims:  # Unflatten array
        i_sc, i_vx, i_s = sim.meta.inds
        sims[i_sc, i_vx, i_s] = sim
        if i_s == 0:
            econdf = sim.get_analyzer(an.econ_analyzer).df
            econdf['location'] = location
            econdf['seed'] = i_s
            econdf['screen_scen'] = sim.meta.vals['screen_scen']
            econdf['vx_scen'] = sim.meta.vals['vx_scen']
            econdfs += econdf
        sim['analyzers'] = []  # Remove the analyzer so we don't need to reduce it
    econ_df = pd.concat(econdfs)
    sc.saveobj(f'{ut.resfolder}/{location}_econ.obj', econ_df)

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_sc in range(len(screen_intvs)):
        for i_vx in range(len(vx_intvs)):
            sim_seeds = sims[i_sc, i_vx, :].tolist()
            all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    msims = np.empty((len(screen_intvs), len(vx_intvs)), dtype=object)
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)
    print('finished making msims')

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_sc, i_vx = msim.meta.inds
        msims[i_sc, i_vx] = msim
        df = pd.DataFrame()
        df['year']                      = msim.results['year']
        df['cancers']                   = msim.results['cancers'][:] # TODO: process in a loop
        df['cancers_low']               = msim.results['cancers'].low
        df['cancers_high']              = msim.results['cancers'].high
        df['asr_cancer_incidence']      = msim.results['asr_cancer_incidence'][:]
        df['asr_cancer_incidence_low']  = msim.results['asr_cancer_incidence'].low
        df['asr_cancer_incidence_high'] = msim.results['asr_cancer_incidence'].high
        df['cancer_deaths']             = msim.results['cancer_deaths'][:]
        df['cancer_deaths_low']         = msim.results['cancer_deaths'].low
        df['cancer_deaths_high']        = msim.results['cancer_deaths'].high
        df['n_vaccinated']              = msim.results['n_vaccinated'][:]
        df['n_vaccinated_low']          = msim.results['n_vaccinated'].low
        df['n_vaccinated_high']         = msim.results['n_vaccinated'].high
        df['location'] = location

        # Store metadata about run
        df['vx_scen'] = msim.meta.vals['vx_scen']
        df['screen_scen'] = msim.meta.vals['screen_scen']
        dfs += df

    alldf = pd.concat(dfs)
    sc.saveobj(f'{ut.resfolder}/{location}.obj', alldf)

    return alldf, msims



#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)

    if 'run_scenarios' in to_run:
        for location in locations:
            if location in ['tanzania', 'myanmar']:
                calib_filestem = '_nov07'
            elif location in ['uganda', 'drc', 'uganda']:
                calib_filestem = '_nov06'
            else:
                calib_filestem = '_nov13'

            # Construct the scenarios
            # Screening scenarios    : No screening, 35% coverage, 70% coverage
            # Vaccine scenarios      : No vaccine, 50% coverage, 90% coverage
            # TxVx                   : No txvx, use case 1 w/ different efficacy values
            screen_scens = sc.objdict({
                'No screening': {},
                # 'HPV, 35% sc cov': dict(
                #     primary='hpv',
                #     screen_coverage=0.35,
                # ),
                # 'HPV, 70% sc cov, 90% tx cov': dict(
                #     primary='hpv',
                #     screen_coverage=0.7,
                #     ltfu=0.1
                # ),
            })

            vx_scens = sc.objdict({
                # 'No vaccine': {},
                'Vx, 50% cov, 9-14': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14)
                ),
                'Vx, 50% cov, 9-14, 90% cov, infant, 90% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                    infant_efficacy=0.9
                ),
                'Vx, 50% cov, 9-14, 90% cov, infant, 70% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                    infant_efficacy=0.7
                ),
                'Vx, 50% cov, 9-14, 90% cov, infant, 50% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                ),
                'Vx, 50% cov, 9-14, 90% cov, infant, 30% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                    infant_efficacy=0.3
                ),

                'Vx, 50% cov, 9-14, 70% cov, infant, 90% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                    infant_coverage=0.7,
                    infant_efficacy=0.9
                ),
                'Vx, 50% cov, 9-14, 70% cov, infant, 70% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                    infant_coverage=0.7,
                    infant_efficacy=0.7
                ),
                'Vx, 50% cov, 9-14, 70% cov, infant, 50% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                    infant_coverage=0.7,
                ),
                'Vx, 50% cov, 9-14, 70% cov, infant, 30% efficacy': dict(
                    vx_coverage=0.5,
                    age_range=(9, 14),
                    infant=True,
                    infant_coverage=0.7,
                    infant_efficacy=0.3
                ),
            })

            alldf, msims = run_scens(screen_intvs=screen_scens, vx_intvs=vx_scens,
                                     location=location, debug=debug, calib_filestem=calib_filestem)

    elif 'plot_scenarios' in to_run:
        locations = [
            # 'india',  # 0
            # 'indonesia',  # 1
            'nigeria',  # 2
            # 'tanzania',  # 3
            # 'bangladesh',  # 4
            # 'myanmar',  # 5
            # 'uganda',  # 6
            # 'ethiopia',  # 7
            # 'drc',  # 8
            # 'kenya'  # 9
        ]

        for location in locations:
            ut.plot_vx_impact(
                location=location,
                scens={
                    'No vaccine': {'vx_scen': 'No vaccine', 'screen_scen': 'No screening'},
                    '50% PxV, 9-14': {'vx_scen': 'Vx, 50% cov, 9-14', 'screen_scen': 'No screening'},
                    '50% PxV, 9-14\n90% infant cov, 70% eff': {'vx_scen': 'Vx, 50% cov, 9-14, 90% cov, infant, 70% efficacy', 'screen_scen': 'No screening'},
                    '50% PxV, 9-14\n90% infant cov, 50% eff': {'vx_scen': 'Vx, 50% cov, 9-14, 90% cov, infant, 50% efficacy', 'screen_scen': 'No screening'},
                    '50% PxV, 9-14\n90% infant cov, 20% eff': {'vx_scen': 'Vx, 50% cov, 9-14, 90% cov, infant, 20% efficacy', 'screen_scen': 'No screening'},
                },
            )



        # ut.plot_CEA(
        #     locations=locations,
        #     background_scens={
        #
        #         '90% PxV, 0% S&T': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
        #         '90% PxV, 35% S&T': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 35% sc cov'},
        #         '90% PxV, 70% S&T': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 70% sc cov, 90% tx cov'},
        #     },
        #
        #     txvx_scen='Mass TxV, 90/50, age 30',
        # )
        #
        #
        #
        #
        #
        print('done')