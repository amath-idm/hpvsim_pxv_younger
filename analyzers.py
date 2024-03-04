"""
Define custom analyzers for HPVsim
TODO: get rid of this file, it's already in HPVsim
"""

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv


class daly_computation(hpv.Analyzer):
    """
    Analyzer for feeding into costing/health economic analysis.
    Produces a dataframe by year storing:
        - Cases/deaths: number of new cancer cases and cancer deaths
        - Average age of new cases, average age of deaths, average age of noncancer death
    """

    def __init__(self, start=2020, life_expectancy=80, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.si = None
        self.df = None
        self.disability_weights = sc.objdict(
            weights=[0.54, 0.049, 0.451, 0.288],
            time_fraction=[0.1, 0.5, 0.3, 0.1],
        )
        self.life_expectancy = life_expectancy
        return

    def av_dw(self):
        dw = self.disability_weights
        len_dw = len(dw.weights)
        return sum([dw.weights[i]*dw.time_fraction[i] for i in range(len_dw)])

    def initialize(self, sim):
        super().initialize(sim)
        columns = ['av_age_cancers', 'av_age_cancer_deaths', 'av_age_other_deaths']
        self.si = sc.findinds(sim.res_yearvec,self.start)[0]
        self.df = pd.DataFrame(0.0, index=pd.Index(sim.res_yearvec[self.si:], name='year'), columns=columns)
        return

    @staticmethod
    def av_age(arr, sim):
        if len(hpv.true(arr)): return np.mean(sim.people.age[hpv.true(arr)])
        else: return np.nan

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start:
            ppl = sim.people
            li = np.floor(sim.yearvec[sim.t])
            lt = (sim.t-1)

            # Age outputs
            self.df.loc[li].av_age_other_deaths = self.av_age(ppl.date_dead_other == lt, sim)
            self.df.loc[li].av_age_cancer_deaths = self.av_age(ppl.date_dead_cancer == lt, sim)
            self.df.loc[li].av_age_cancers = self.av_age(ppl.date_cancerous == lt, sim)
        return

    def finalize(self, sim):
        # Add in results that are already generated (NB, these have all been scaled already)
        self.df['cancers'] = sim.results['cancers'][self.si:]
        self.df['cancer_deaths'] = sim.results['cancer_deaths'][self.si:]
        self.df['other_deaths'] = sim.results['other_deaths'][self.si:]
        self.df['cancer_years'] = self.df['av_age_cancer_deaths'] - self.df['av_age_cancers']

        # Actually compute DALYs
        self.df['yld'] = self.av_dw() * self.df['cancer_years'] * self.df['cancers']
        self.df['yll'] = (self.life_expectancy - self.df['av_age_cancer_deaths']) * self.df['cancer_deaths']
        self.df['dalys'] = self.df['yld'] + self.df['yll']

        return

