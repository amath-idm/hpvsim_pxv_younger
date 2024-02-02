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
    'india',        # 0
    # 'indonesia',    # 1
    # 'nigeria',      # 2
    # 'tanzania',     # 3
    # 'bangladesh',   # 4
    # 'myanmar',      # 5
    # 'uganda',       # 6
    # 'ethiopia',     # 7
    # 'drc',          # 8
    # 'kenya'         # 9
]

debug = False
n_seeds = [3, 1][debug] # How many seeds to run per cluster

#%% Functions

def make_msims(sims, use_mean=True):
    '''
    Utility to take a slice of sims and turn it into a multisim
    '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_sc, i_vx, i_s, i_p = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except parameter set matches
        assert i_sc == sim.meta.inds[0]
        assert i_vx == sim.meta.inds[1]
        # assert (s == 0) or i_s != sim.meta.inds[2]
        # assert i_p == sim.meta.inds[3]
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
    calib_num = [10, 2][debug]
    calib_pars_list = sc.loadobj(f'results/{location}_pars{calib_filestem}.obj')[:calib_num]
    n_calib = len(calib_pars_list)
    n_sims = len(screen_intvs) * len(vx_intvs) * n_seeds * n_calib

    for i_sc, sc_label, screen_scen_pars in screen_intvs.enumitems():
        for i_vx, vx_label, vx_scen_pars in vx_intvs.enumitems():
            for i_s in range(n_seeds):  # n samples per cluster
                for i_p, calib_pars in enumerate(calib_pars_list):
                    count += 1
                    meta = sc.objdict()
                    meta.count = count
                    meta.n_sims = n_sims
                    meta.inds = [i_sc, i_vx, i_s, i_p]
                    calib_pars.pop('hiv_pars', None)
                    meta.vals = sc.objdict(sc.mergedicts(screen_scen_pars, vx_scen_pars, calib_pars,
                                                        dict(seed=i_s, screen_scen=sc_label,
                                                            vx_scen=vx_label, calib_par = i_p)))
                    ikw.append(sc.objdict(screen_intv=screen_scen_pars, vx_intv=vx_scen_pars,
                                        seed=i_s, calib_pars = calib_pars))
                    ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    n_workers = 8
    n_agents = [50e3, 1e3][debug]
    kwargs = dict(verbose=verbose, debug=debug, location=location,
                  econ_analyzer=True, end=2100, n_agents=n_agents)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs, ncpus=n_workers)

    # Rearrange sims
    sims = np.empty((len(screen_intvs), len(vx_intvs), n_seeds, n_calib), dtype=object)
    econdfs = sc.autolist()
    for sim in all_sims:  # Unflatten array
        i_sc, i_vx, i_s, i_p = sim.meta.inds
        sims[i_sc, i_vx, i_s, i_p] = sim
        if i_s == 0:
            econdf = sim.get_analyzer(an.econ_analyzer).df
            econdf['location'] = location
            econdf['seed'] = i_s
            econdf['calib_param'] = i_p
            econdf['screen_scen'] = sim.meta.vals['screen_scen']
            econdf['vx_scen'] = sim.meta.vals['vx_scen']
            econdfs += econdf
        sim['analyzers'] = []  # Remove the analyzer so we don't need to reduce it
    econ_df = pd.concat(econdfs)
    sc.saveobj(f'{ut.resfolder}/{location}_econ{calib_filestem}.obj', econ_df)

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_sc in range(len(screen_intvs)):
        for i_vx in range(len(vx_intvs)):
            sim_seeds = sims[i_sc, i_vx, :,:].flatten().tolist()
            all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    msims = np.empty((len(screen_intvs), len(vx_intvs)), dtype=object)
    
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)
    msim_results = sc.objdict()
    for msim in all_msims:
        i_sc, i_vx = msim.meta.inds
        key = f'{i_sc}_{i_vx}'
        msim_results[key] = sc.objdict()
        msim_results[key]['location'] = location
        msim_results[key]['meta'] = msim.meta
        msim_results[key]['results'] = msim.results
    sc.saveobj(f'{ut.resfolder}/{location}_scenarios{calib_filestem}.obj', msim_results)
    return msim_results

def organize_msim_results(msim_results, file_loc):
    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    output_list = ['cancers','asr_cancer_incidence','cancer_deaths',
                   'n_vaccinated','infections','n_infectious','n_precin','n_cin','n_susceptible',
                   'hpv_prevalence_by_age','reactivations_by_age','reinfections_by_age','infections_by_age',
                   'cancer_incidence_by_age','cancers_by_age','hpv_incidence_by_age','n_infectious_by_age']
    for key, val in msim_results.items():
        i_sc, i_vx = val.meta.inds
        df = pd.DataFrame()
        df['year']  = val.results['year']
        df['location'] = location
        df['vx_scen'] = val.meta.vals['vx_scen']
        df['screen_scen'] = val.meta.vals['screen_scen']
        for output in output_list:
            mean = val.results[output][:]
            if mean.ndim >1:
                for k in range(mean.shape[0]):
                    df[output+'_'+str(k)] =  val.results[output][k,:]
                    df[output+'_'+str(k)+'_low'] =  val.results[output].low[k,:]
                    df[output+'_'+str(k)+'_high'] =  val.results[output].high[k,:]
            else:
                df[output] = val.results[output][:]
                df[output+'_low'] = val.results[output].low
                df[output+'_high'] = val.results[output].high
        dfs += df
    alldf = pd.concat(dfs, axis=0)
    alldf.to_csv(f'{ut.resfolder}/{file_loc}_df.csv')
    return alldf



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
                calib_filestem = '_jan10_hpv_control_100'
                # calib_filestem = '_calib50_lowact'

            # Construct the scenarios
            # Screening scenarios    : No screening, 35% coverage, 70% coverage
            # Vaccine scenarios      : No vaccine, 50% coverage, 90% coverage
            # TxVx                   : No txvx, use case 1 w/ different efficacy values
            screen_scens = sc.objdict({
                'HPV, 15% sc cov': dict(
                    primary='hpv',
                    screen_coverage=0.15,
                ),
            })

            vx_scens = sc.objdict({
                'No vaccine': {},
                'Vx age 9-15, 90% cov': dict(
                    age_range=(9, 15),
                ),
                'Vx age 9-25, 90% cov': dict(
                    age_range=(9, 25),
                ),
                'Vx age 9-35, 90% cov': dict(
                    age_range=(9, 35),
                ),
            })
            # run_scens outputs obj of msims.results file
            # msim_results = run_scens(screen_intvs=screen_scens, vx_intvs=vx_scens,
            #                          location=location, debug=debug, calib_filestem=calib_filestem)
            # or read from obj file
            filename = f'{location}_scenarios{calib_filestem}'
            msim_results = sc.load(f'results/{filename}.obj')
            alldf = organize_msim_results(msim_results, filename)
            
            # alldf, msims = run_scens(screen_intvs=screen_scens, vx_intvs=vx_scens,
            #                          location=location, debug=debug, calib_filestem=calib_filestem)

    elif 'plot_scenarios' in to_run:
        locations = [
            'india',  # 0
            # 'indonesia',  # 1
            # 'nigeria',  # 2
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
                background_scen={'screen_scen': 'HPV, 15% sc cov'},
                adolescent_coverages=[20, 40, 60],
                infant_efficacies=[0],
                # infant_coverage=90
            )

            ut.plot_CEA(
                location=location,
                background_scen={'screen_scen': 'HPV, 15% sc cov'},
                adolescent_coverages=[20, 40, 60],
                # infant_efficacies=[50, 70, 90],
                # infant_coverage=90
            )

            ut.plot_resource_use(
                location=location,
                background_scen={'screen_scen': 'HPV, 15% sc cov'},
                adolescent_coverages=[20, 40, 60],
                # infant_efficacies=[50, 70, 90],
                # infant_coverage=90
            )


        print('done')
# %%
