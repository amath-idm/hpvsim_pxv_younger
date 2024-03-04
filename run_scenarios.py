"""
Run HPVsim scenarios varying the age of prophylactic vaccination
Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
"""


# %% General settings

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
import run_sim as rs

# Settings - used here and imported elsewhere
debug = 0
n_seeds = [3, 1][debug]  # How many seeds to run per cluster
coverage_arr = [0.3, 0.5, 0.7, 0.9]
efficacy_arr = [0.5, 0.6, 0.7, 0.8, 0.9]


# %% Create interventions

def make_st(screen_coverage=0.15, treat_coverage=0.7, start_year=2020):
    """ Make screening & treatment intervention """

    age_range = [30, 50]
    len_age_range = (age_range[1]-age_range[0])/2
    model_annual_screen_prob = 1 - (1 - screen_coverage)**(1/len_age_range)

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                  (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    screening = hpv.routine_screening(
        prob=model_annual_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        product='hpv',
        age_range=age_range,
        label='screening'
    )

    # Assign treatment
    screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
    assign_treatment = hpv.routine_triage(
        start_year=start_year,
        prob=1.0,
        annual_prob=False,
        product='tx_assigner',
        eligibility=screen_positive,
        label='tx assigner'
    )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob=treat_coverage,
        annual_prob=False,
        product='ablation',
        eligibility=ablation_eligible,
        label='ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() +
                                             sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(
        prob=treat_coverage,
        annual_prob=False,
        product='excision',
        eligibility=excision_eligible,
        label='excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_num(
        prob=treat_coverage/4,  # assume an additional dropoff in CaTx coverage
        annual_prob=False,
        product=hpv.radiation(),
        eligibility=radiation_eligible,
        label='radiation'
    )

    st_intvs = [screening, assign_treatment, ablation, excision, radiation]

    return st_intvs


def make_vx_scenarios(coverage_arr, efficacy_arr, product='nonavalent', start_year=2025):

    age_range = (9, 14)
    catchup_age = (age_range[0]+1, age_range[1])
    routine_age = (age_range[0], age_range[0]+1)
    prod = hpv.default_vx(prod_name=product)
    eligibility = lambda sim: (sim.people.doses == 0)

    vx_scenarios = dict()

    # Baseline
    vx_scenarios['Baseline'] = []

    # Construct the adolescent only scenarios
    for cov_val in coverage_arr:
        label = f'Adolescent: {cov_val} coverage'
        routine_vx = hpv.routine_vx(
            prob=cov_val,
            start_year=start_year,
            product=prod,
            age_range=routine_age,
            eligibility=eligibility,
            label='Routine vx'
        )

        catchup_vx = hpv.campaign_vx(
            prob=cov_val,
            years=start_year,
            product=prod,
            age_range=catchup_age,
            eligibility=eligibility,
            label='Catchup vx'
        )

        vx_scenarios[label] = [routine_vx, catchup_vx]

    # Construct the infant scenarios
    for cov_val in coverage_arr:
        for eff_val in efficacy_arr:

            label = f'Adolescents: {cov_val} coverage, Infants: {eff_val} efficacy'

            routine_vx = hpv.routine_vx(
                prob=cov_val,
                years=[start_year, start_year+9],
                product=prod,
                age_range=routine_age,
                eligibility=eligibility,
                label='Routine vx'
            )

            catchup_vx = hpv.campaign_vx(
                prob=cov_val,
                years=start_year,
                product=prod,
                age_range=catchup_age,
                eligibility=eligibility,
                label='Catchup vx'
            )

            infant_prod = hpv.default_vx(prod_name=product)
            infant_prod.imm_init = dict(dist='beta_mean', par1=eff_val, par2=0.025)
            infant_vx = hpv.routine_vx(
                prob=0.9,
                start_year=start_year,
                product=infant_prod,
                age_range=(0, 1),
                label='Infant vx'
            )

            these_intvs = [routine_vx, catchup_vx, infant_vx]
            vx_scenarios[label] = these_intvs

    return vx_scenarios


def make_sims(calib_pars=None, vx_scenarios=None):
    """ Set up scenarios """

    st_intv = make_st()

    all_msims = sc.autolist()
    for name, vx_intv in vx_scenarios.items():
        sims = sc.autolist()
        for seed in range(n_seeds):
            interventions = vx_intv + st_intv
            sim = rs.make_sim(calib_pars=calib_pars, debug=debug, interventions=interventions, end=2100, seed=seed)
            sim.label = name
            sims += sim
        all_msims += hpv.MultiSim(sims)

    msim = hpv.MultiSim.merge(all_msims, base=False)

    return msim


def run_sims(calib_pars=None, vx_scenarios=None, verbose=0.2):
    """ Run the simulations """
    msim = make_sims(calib_pars=calib_pars, vx_scenarios=vx_scenarios)
    msim.run(verbose=verbose)
    return msim


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    do_run = True
    do_save = False
    do_process = True

    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)
    if do_run:
        calib_pars = sc.loadobj('results/nigeria_pars_nov13.obj')
        vx_scenarios = make_vx_scenarios(coverage_arr, efficacy_arr)
        msim = run_sims(calib_pars=calib_pars, vx_scenarios=vx_scenarios)

        if do_save: msim.save('results/vs.msim')

        if do_process:

            metrics = ['year', 'asr_cancer_incidence', 'cancers', 'cancer_deaths']

            # Process results
            vx_scenarios = make_vx_scenarios(coverage_arr, efficacy_arr)
            scen_labels = list(vx_scenarios.keys())
            mlist = msim.split(chunks=len(scen_labels))

            msim_dict = sc.objdict()
            for si, scen_label in enumerate(scen_labels):
                reduced_sim = mlist[si].reduce(output=True)
                mres = sc.objdict({metric: reduced_sim.results[metric] for metric in metrics})
                mres['dalys'] = reduced_sim.get_analyzer().df

                for ii, intv in enumerate(reduced_sim['interventions']):
                    intv_label = intv.label
                    mres[intv_label] = reduced_sim['interventions'][ii].n_products_used

                msim_dict[scen_label] = mres

            sc.saveobj(f'results/vx_scens.obj', msim_dict)

    print('Done.')
