'''
Define custom intervention
'''

import hpvsim as hpv
from hpvsim import utils as hpu
from hpvsim import defaults as hpd
from hpvsim import immunity as hpimm
import numpy as np


class TxVx(hpv.tx):
    def __init__(self, decay_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate

        self.map = {
            'latent': 'date_exposed',
            'precin': 'date_infectious',
            'cin': 'date_cin',
            'cancerous': 'date_cin',
        }

    def administer(self, sim, inds, return_format='dict'):
        '''
        Loop over treatment states to determine those who are successfully treated and clear infection
        '''

        tx_successful = []  # Initialize list of successfully treated individuals
        people = sim.people
        decay_rate = float(self.decay_rate)

        for state in self.states:  # Loop over states
            for g, genotype in sim['genotype_map'].items():  # Loop over genotypes in the sim

                theseinds = inds[hpu.true(people[state][g, inds])]  # Extract people for whom this state is true for this genotype

                if len(theseinds):
                    attr = self.map[state]
                    t_in_state = (sim.people.t - sim.people[attr][g, theseinds]) / sim.pars['dt']
                    decay = (1 - decay_rate) ** t_in_state
                    df_filter = (self.df.state == state)  # Filter by state
                    if self.ng > 1: df_filter = df_filter & (self.df.genotype == genotype)
                    thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype

                    # Determine whether treatment is successful
                    efficacy = thisdf.efficacy.values[0]*decay
                    eff_probs = np.full(len(theseinds), efficacy, dtype=hpd.default_float)  # Assign probabilities of treatment success
                    to_eff_treat = hpu.binomial_arr(eff_probs)  # Determine who will have effective treatment
                    eff_treat_inds = theseinds[to_eff_treat]
                    if len(eff_treat_inds):
                        tx_successful += list(eff_treat_inds)
                        people[state][g, eff_treat_inds] = False  # People who get treated have their CINs removed
                        people['cin'][g, eff_treat_inds] = False  # People who get treated have their CINs removed
                        people[f'date_{state}'][g, eff_treat_inds] = np.nan
                        people[f'date_cancerous'][g, eff_treat_inds] = np.nan

                        # alternatively, clear now:
                        people.susceptible[g, eff_treat_inds] = True
                        people.infectious[g, eff_treat_inds] = False
                        people.inactive[g, eff_treat_inds] = False  # should already be false
                        hpimm.update_peak_immunity(people, eff_treat_inds, imm_pars=people.pars, imm_source=g)  # update immunity
                        people.date_reactivated[g, eff_treat_inds] = np.nan

        tx_successful = np.array(list(set(tx_successful)))
        tx_unsuccessful = np.setdiff1d(inds, tx_successful)
        if return_format == 'dict':
            output = {'successful': tx_successful, 'unsuccessful': tx_unsuccessful}
        elif return_format == 'array':
            output = tx_successful

        if self.imm_init is not None:
            people.cell_imm[self.imm_source, inds] = hpu.sample(**self.imm_init, size=len(inds))
            people.t_imm_event[self.imm_source, inds] = people.t
        elif self.imm_boost is not None:
            people.cell_imm[self.imm_source, inds] *= self.imm_boost
            people.t_imm_event[self.imm_source, inds] = people.t
        return output
