"""
Define custom analyzers for HPVsim
"""

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv


class econ_analyzer(hpv.Analyzer):
    """
    Analyzer for feeding into costing/health economic analysis.
    Produces a dataframe by year storing:
        
        - Resource use: number of vaccines, screens, lesions treated, cancers treated
        - Cases/deaths: number of new cancer cases and cancer deaths
        - Average age of new cases, average age of deaths, average age of noncancer death
    """

    def __init__(self, start=2020, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.si = None
        self.df = None
        return

    def initialize(self, sim):
        super().initialize(sim)
        columns = ['new_hpv_screens', 'new_vaccinations', 'new_infant_vaccinations',
                   'new_thermal_ablations', 'new_leeps', 'new_cancer_treatments',
                   'new_cancers', 'new_cancer_deaths', 'new_other_deaths',
                   'av_age_cancers', 'av_age_cancer_deaths', 'av_age_other_deaths']
        self.si = sc.findinds(sim.res_yearvec,self.start)[0]
        self.df = pd.DataFrame(0.0, index=pd.Index(sim.res_yearvec[self.si:], name='year'), columns=columns)
        return

    @staticmethod
    def av_age(arr, sim):
        if len(hpv.true(arr)): return np.mean(sim.people.age[hpv.true(arr)])
        else: return np.nan

    def apply(self, sim):
        if sim.yearvec[sim.t]>=self.start:
            ppl = sim.people
            li = np.floor(sim.yearvec[sim.t])
            ltt = int((sim.t-1)*sim['dt'])
            lt = (sim.t-1)

            # Pull out characteristics of sim to decide what resources we need
            simvals = sim.meta.vals
            screen_scen_label = simvals.screen_scen
            vx_scen_label = simvals.vx_scen
            if screen_scen_label != 'No screening':
                # Resources
                self.df.loc[li].new_hpv_screens += sim.get_intervention('screening').n_products_used.values[ltt]
                self.df.loc[li].new_thermal_ablations += sim.get_intervention('ablation').n_products_used.values[ltt]
                self.df.loc[li].new_leeps += sim.get_intervention('excision').n_products_used.values[ltt]
                self.df.loc[li].new_cancer_treatments += sim.get_intervention('radiation').n_products_used.values[ltt]

            if vx_scen_label != 'No vaccine':
                # Resources
                self.df.loc[li].new_vaccinations += sim.get_intervention('Routine vx').n_products_used.values[ltt]
                self.df.loc[li].new_vaccinations += sim.get_intervention('Catchup vx').n_products_used.values[ltt]

                if 'infant' in vx_scen_label:
                    self.df.loc[li].new_infant_vaccinations += sim.get_intervention('Infant vx').n_products_used.values[ltt]

            # Age outputs
            self.df.loc[li].av_age_other_deaths = self.av_age(ppl.date_dead_other == lt)
            self.df.loc[li].av_age_cancer_deaths = self.av_age(ppl.date_dead_cancer == lt)
            self.df.loc[li].av_age_cancers = self.av_age(ppl.date_cancerous == lt)
        return

    def finalize(self, sim):
        # Add in results that are already generated (NB, these have all been scaled already)
        self.df['new_cancers'] = sim.results['cancers'][self.si:]
        self.df['new_cancer_deaths'] = sim.results['cancer_deaths'][self.si:]
        self.df['new_other_deaths'] = sim.results['other_deaths'][self.si:]
        return

