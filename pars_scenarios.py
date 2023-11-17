'''
Define parameters used in scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import pars_data as dp
import interventions as hpi


def get_vx_intvs(start_year=2025, vx_coverage=0.9, age_range=(9,14), product='nonavalent', infant=False,
                 infant_efficacy=0.5, infant_coverage=0.9):

    catchup_age = (age_range[0]+1, age_range[1])
    routine_age = (age_range[0], age_range[0]+1)
    prod = hpv.default_vx(prod_name=product)
    eligibility = lambda sim: (sim.people.vx_doses == 0)

    routine_vx = hpv.routine_vx(
        prob=vx_coverage,
        start_year=start_year,
        product=prod,
        age_range=routine_age,
        eligibility=eligibility,
        label='Routine vx'
    )

    catchup_vx = hpv.campaign_vx(
        prob=vx_coverage,
        years=start_year,
        product=prod,
        age_range=catchup_age,
        eligibility=eligibility,
        label='Catchup vx'
    )

    if infant:
        infant_prod = hpv.default_vx(prod_name=product)
        infant_prod.imm_init = dict(dist='beta_mean', par1=infant_efficacy, par2=0.025)
        infant_vx = hpv.routine_vx(
            prob=infant_coverage,
            start_year=start_year,
            product=infant_prod,
            age_range=(0,1),
            label='Infant vx'
        )
        return [routine_vx, catchup_vx, infant_vx]
    else:
        return [routine_vx, catchup_vx]

def get_screen_intvs(primary=None, triage=None, screen_coverage=0.7, ltfu=0.3, start_year=2025):
    '''
    Make interventions for screening scenarios

    primary (None or dict): dict of test positivity values for precin, cin1, cin2, cin3 and cancerous
    triage (None or dict): dict of test positivity values for precin, cin1, cin2, cin3 and cancerous
    '''

    # Return empty list if nothing is defined
    if primary is None: return []

    # Create screen products
    if isinstance(primary, str):
        primary_test = hpv.default_dx(prod_name=primary)
    elif isinstance(primary, dict):
        primary_test = make_screen_test(**primary)
    if triage is not None:
        if isinstance(triage, str):
            triage_test = hpv.default_dx(prod_name=triage)
        elif isinstance(triage, dict):
            triage_test = make_screen_test(**triage)

    tx_assigner = make_tx_assigner()

    age_range = [30,50]
    len_age_range = (age_range[1]-age_range[0])/2

    model_annual_screen_prob = 1 - (1 - screen_coverage)**(1/len_age_range)

    # Routine screening
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                  (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    screening = hpv.routine_screening(
        product=primary_test,
        prob=model_annual_screen_prob,
        eligibility=screen_eligible,
        age_range=[30, 50],
        start_year=start_year,
        label='screening'
    )

    if triage is not None:
        # Triage screening
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1 - ltfu,
            annual_prob=False,
            product=triage_test,
            eligibility=screen_positive,
            label='triage'
        )
        triage_positive = lambda sim: sim.get_intervention('triage').outcomes['positive']
        assign_treatment = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product=tx_assigner,
            eligibility=triage_positive,
            label='tx assigner'
        )
    else:
        # Assign treatment
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product=tx_assigner,
            eligibility=screen_positive,
            label='tx assigner'
        )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob=1-ltfu,
        annual_prob=False,
        product='ablation',
        eligibility=ablation_eligible,
        label='ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() +
                                             sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(
        prob=1-ltfu,
        annual_prob=False,
        product='excision',
        eligibility=excision_eligible,
        label='excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_num(
        prob=(1-ltfu)/4, # assume an additional dropoff in CaTx coverage
        annual_prob=False,
        product=hpv.radiation(),
        eligibility=radiation_eligible,
        label='radiation'
    )

    if triage is not None:
        st_intvs = [screening, triage_screening, assign_treatment, ablation, excision, radiation]
    else:
        st_intvs = [screening, triage_screening, ablation, excision, radiation]

    return st_intvs


def make_screen_test(precin=0.25, cin=0.45, cancerous=0.6):
    '''
    Make screen product using P(T+| health state) for health states HPV, CIN1, CIN2, CIN3, and cancer
    '''

    basedf = pd.read_csv('dx_pars.csv')
    not_changing_states = ['susceptible', 'latent']
    not_changing = basedf.loc[basedf.state.isin(not_changing_states)].copy()

    new_states = sc.autolist()
    for state, posval in zip(['precin', 'cin', 'cancerous'],
                             [precin, cin, cancerous]):
        new_pos_vals = basedf.loc[(basedf.state == state) & (basedf.result == 'positive')].copy()
        new_pos_vals.probability = posval
        new_neg_vals = basedf.loc[(basedf.state == state) & (basedf.result == 'negative')].copy()
        new_neg_vals.probability = 1-posval
        new_states += new_pos_vals
        new_states += new_neg_vals
    new_states_df = pd.concat(new_states)

    # Make the screen product
    screen_test = hpv.dx(pd.concat([not_changing, new_states_df]), hierarchy=['positive', 'negative'])
    return screen_test

def make_tx_assigner():
    '''
    Make treatment assigner product
    '''

    basedf = pd.read_csv('tx_assigner_pars.csv')
    # Make the screen product
    screen_test = hpv.dx(basedf, hierarchy=['ablation', 'excision', 'radiation'])
    return screen_test