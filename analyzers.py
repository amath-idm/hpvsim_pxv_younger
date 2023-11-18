'''
Define custom analyzers for HPVsim
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import math


class AFS(hpv.Analyzer):
    def __init__(self, bins=None, cohort_starts=None, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins or np.arange(12,31,1)
        self.cohort_starts = cohort_starts
        self.binspan = self.bins[-1]-self.bins[0]

    def initialize(self, sim):
        super().initialize()
        if self.cohort_starts is None:
            first_cohort = sim['start'] + sim['burnin'] - 5
            last_cohort = sim['end']-self.binspan
            self.cohort_starts = sc.inclusiverange(first_cohort, last_cohort)
            self.cohort_ends = self.cohort_starts+self.binspan
            self.n_cohorts = len(self.cohort_starts)
            self.cohort_years = np.array([sc.inclusiverange(i,i+self.binspan) for i in self.cohort_starts])

        self.prop_active_f = np.zeros((self.n_cohorts,self.binspan+1))
        self.prop_active_m = np.zeros((self.n_cohorts,self.binspan+1))

    def apply(self, sim):
        if sim.yearvec[sim.t] in self.cohort_years:
            cohort_inds, bin_inds = sc.findinds(self.cohort_years, sim.yearvec[sim.t])
            for ci,cohort_ind in enumerate(cohort_inds):
                bin_ind = bin_inds[ci]
                bin = self.bins[bin_ind]

                conditions_f = sim.people.is_female * sim.people.alive * (sim.people.age >= (bin-1)) * (sim.people.age < bin) * sim.people.level0
                denom_inds_f = hpv.true(conditions_f)
                num_conditions_f = conditions_f * (sim.people.n_rships.sum(axis=0)>0)
                num_inds_f = hpv.true(num_conditions_f)
                self.prop_active_f[cohort_ind,bin_ind] = len(num_inds_f)/len(denom_inds_f)

                conditions_m = ~sim.people.is_female * sim.people.alive * (sim.people.age >= (bin-1)) * (sim.people.age < bin)# * sim.people.level0
                denom_inds_m = hpv.true(conditions_m)
                num_conditions_m = conditions_m * (sim.people.n_rships.sum(axis=0)>0)
                num_inds_m = hpv.true(num_conditions_m)
                self.prop_active_m[ci,bin_ind] = len(num_inds_m)/len(denom_inds_m)
        return
    

class cum_dist(hpv.Analyzer):
    '''
    Determine distribution of time to clearance, persistence, pre-cancer and cancer
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year

    def initialize(self, sim):
        super().initialize(sim)
        if self.start_year is None:
            self.start_year = sim['start']
        self.dur_to_clearance = []
        self.dur_to_cin = []
        self.dur_to_cancer = []
        self.total_infections = 0


    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            inf_genotypes, inf_inds = (sim.people.date_exposed == sim.t).nonzero()
            self.total_infections += len(inf_inds)
            if len(inf_inds):
                infs_that_progress_bools = hpv.utils.defined(sim.people.date_cin[inf_genotypes, inf_inds])
                infs_that_progress_inds = hpv.utils.idefined(sim.people.date_cin[inf_genotypes, inf_inds], inf_inds)
                infs_to_cancer_bools = hpv.utils.defined(sim.people.date_cancerous[inf_genotypes, inf_inds])
                infs_to_cancer_inds = hpv.utils.idefined(sim.people.date_cancerous[inf_genotypes, inf_inds], inf_inds)
                infs_that_clear_bools = hpv.utils.defined(sim.people.date_clearance[inf_genotypes, inf_inds])
                infs_that_clear_inds = hpv.utils.idefined(sim.people.date_clearance[inf_genotypes, inf_inds], inf_inds)

                dur_to_clearance = (sim.people.date_clearance[inf_genotypes[infs_that_clear_bools], infs_that_clear_inds] - sim.t)*sim['dt']
                dur_to_cin = (sim.people.date_cin[inf_genotypes[infs_that_progress_bools], infs_that_progress_inds] - sim.t)*sim['dt']
                dur_to_cancer = (sim.people.date_cancerous[inf_genotypes[infs_to_cancer_bools], infs_to_cancer_inds] - sim.t)*sim['dt']

                self.dur_to_clearance += dur_to_clearance.tolist()
                self.dur_to_cin += dur_to_cin.tolist()
                self.dur_to_cancer += dur_to_cancer.tolist()

class prop_cleared(hpv.Analyzer):
    '''
    Determine the percentage of cleared infections over time.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year

    def initialize(self, sim):
        super().initialize(sim)
        if self.start_year is None:
            self.start_year = sim['start']
        self.n_years = 20
        self.num_cleared = np.zeros((sim['n_genotypes'], int(self.n_years/sim['dt'])))
        self.tvec = np.arange(0, self.n_years, sim['dt'])
        self.num_infections = np.zeros(sim['n_genotypes'])

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            for gtype in range(sim['n_genotypes']):
                inf_inds = (sim.people.date_exposed[gtype,:] == sim.t).nonzero()[0]

                self.num_infections[gtype] += len(inf_inds)
                if len(inf_inds):
                    date_clearance = sim.people.date_clearance[gtype, inf_inds]
                    total_time = (date_clearance - sim.t) * sim['dt']
                    time_to_clear, num = np.unique(total_time[~np.isnan(total_time)], return_counts=True)
                    time_to_clear_adjusted = time_to_clear[time_to_clear < self.n_years]
                    num[len(time_to_clear_adjusted)-1] += np.sum(num[len(time_to_clear_adjusted)-1:])
                    num = num[:len(time_to_clear_adjusted)]
                    inds = np.digitize(time_to_clear_adjusted, self.tvec)-1
                    self.num_cleared[gtype, inds] += num

        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''

    @staticmethod
    def reduce(analyzers, quantiles=None):
        if quantiles is None: quantiles = {'low': 0.1, 'high': 0.9}
        base_az = analyzers[0]
        reduced_az = sc.dcp(base_az)
        reduced_az.num_cleared = dict()
        reduced_az.num_infections = dict()
        for ng in range(len(base_az.num_infections)):
            reduced_az.num_cleared[ng] = sc.objdict()
            reduced_az.num_infections[ng] = sc.objdict()
            allres_cleared = np.empty([len(analyzers), base_az.num_cleared.shape[1]])
            allres_infections = np.empty([len(analyzers)])
            for ai,az in enumerate(analyzers):
                allres_cleared[ai,:] = az.num_cleared[ng,:]
                allres_infections[ai] = az.num_infections[ng]
            reduced_az.num_cleared[ng].best  = np.quantile(allres_cleared, 0.5, axis=0)
            reduced_az.num_cleared[ng].low   = np.quantile(allres_cleared, quantiles['low'], axis=0)
            reduced_az.num_cleared[ng].high  = np.quantile(allres_cleared, quantiles['high'], axis=0)

            reduced_az.num_infections[ng].best  = np.quantile(allres_infections, 0.5)
            reduced_az.num_infections[ng].low   = np.quantile(allres_infections, quantiles['low'])
            reduced_az.num_infections[ng].high  = np.quantile(allres_infections, quantiles['high'])

        return reduced_az
class age_causal_by_time(hpv.Analyzer):
    '''
    Determine the age at which people with cervical cancer were causally infected and
    time spent between infection and cancer.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        for gtype in range(sim['n_genotypes']):
            self.age_causal = np.zeros(len(self.years))
            self.age_cancer = np.zeros(len(self.years))
            self.dwelltime = np.zeros(len(self.years))

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                total_time = (sim.t - date_exposed) * sim['dt']
                self.age_causal[sim.t] = np.median((current_age - total_time))
                self.age_cancer[sim.t] = np.median(current_age)
                self.dwelltime[sim.t] = np.median(total_time)
        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''

class prop_exposed(hpv.Analyzer):
    ''' Store proportion of agents exposed '''
    def __init__(self, years=None):
        super().__init__()
        self.years = years
        self.timepoints = []

    def initialize(self, sim):
        super().initialize(sim)
        for y in self.years:
            try:    tp = sc.findinds(sim.yearvec, y)[0]
            except: raise ValueError('Year not found')
            self.timepoints.append(tp)
        self.prop_exposed = dict()
        for y in self.years: self.prop_exposed[y] = []

    def apply(self, sim):
        if sim.t in self.timepoints:
            tpi = self.timepoints.index(sim.t)
            year = self.years[tpi]
            prop_exposed = sc.autolist()
            for a in range(10,30):
                ainds = hpv.true((sim.people.age >= a) & (sim.people.age < a+1) & (sim.people.sex==0))
                prop_exposed += sc.safedivide(sum((~np.isnan(sim.people.date_exposed[:, ainds])).any(axis=0)), len(ainds))
            self.prop_exposed[year] = np.array(prop_exposed)
        return

    @staticmethod
    def reduce(analyzers, quantiles=None):
        if quantiles is None: quantiles = {'low': 0.1, 'high': 0.9}
        base_az = analyzers[0]
        reduced_az = sc.dcp(base_az)
        reduced_az.prop_exposed = dict()
        for year in base_az.years:
            reduced_az.prop_exposed[year] = sc.objdict()
            allres = np.empty([len(analyzers), len(base_az.prop_exposed[year])])
            for ai,az in enumerate(analyzers):
                allres[ai,:] = az.prop_exposed[year][:]
            reduced_az.prop_exposed[year].best  = np.quantile(allres, 0.5, axis=0)
            reduced_az.prop_exposed[year].low   = np.quantile(allres, quantiles['low'], axis=0)
            reduced_az.prop_exposed[year].high  = np.quantile(allres, quantiles['high'], axis=0)

        return reduced_az

class latency(hpv.Analyzer):
    '''
    Determine the age at which people with cervical cancer were causally infected and
    time spent between infection and cancer.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.age_causal = dict()
        self.age_cancer = dict()
        for latency in ['None', 'Latent']:
            self.age_causal[latency] = []
            self.age_cancer[latency] = []
        self.age_latent = []
        self.age_reactivated = []


    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                date_latent = sim.people.date_latent[cancer_genotypes, cancer_inds]
                date_reactivated = sim.people.date_reactivated[cancer_genotypes, cancer_inds]
                latent_inds = hpv.defined(date_latent)
                not_latent_inds = hpv.undefined(date_latent)
                time_since_reactivation = (sim.t - date_reactivated[latent_inds]) * sim['dt']
                time_since_latent = (sim.t - date_latent[latent_inds]) * sim['dt']
                total_time = (sim.t - date_exposed) * sim['dt']

                self.age_causal['Latent'] += (current_age[latent_inds] - total_time[latent_inds]).tolist()
                self.age_causal['None'] += (current_age[not_latent_inds] - total_time[not_latent_inds]).tolist()
                self.age_cancer['Latent'] += current_age[latent_inds].tolist()
                self.age_cancer['None'] += current_age[not_latent_inds].tolist()
                self.age_reactivated += (current_age[latent_inds] - time_since_reactivation).tolist()
                self.age_latent += (current_age[latent_inds] - time_since_latent).tolist()

        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''

class dwelltime_by_genotype(hpv.Analyzer):
    '''
    Determine the age at which people with cervical cancer were causally infected and
    time spent between infection and cancer.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.age_causal = []
        self.age_cancer = []
        self.age_cin = []


    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                dur_cin = sim.people.dur_cin[cancer_genotypes, cancer_inds]
                total_time = (sim.t - date_exposed) * sim['dt']
                self.age_causal += (current_age - total_time).tolist()
                self.age_cin += (current_age - dur_cin).tolist()
                self.age_cancer += (current_age).tolist()
        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''
        return


class dwelltime_by_latency(hpv.Analyzer):
    '''
    Determine the age at which people with cervical cancer were causally infected and
    time spent between infection and cancer.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.age_causal = dict()
        self.age_cancer = dict()
        self.dwelltime = dict()
        for latency in ['None', 'Latent']:
            self.age_causal[latency] = []
            self.age_cancer[latency] = []
        for state in ['precin', 'cin1', 'cin2', 'cin3', 'total']:
            self.dwelltime[state] = dict()
            for latency in ['None', 'Latent']:
                self.dwelltime[state][latency] = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_infectious = sim.people.date_infectious[cancer_genotypes, cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                date_latent = sim.people.date_latent[cancer_genotypes, cancer_inds]
                date_cin1 = sim.people.date_cin1[cancer_genotypes, cancer_inds]
                date_cin2 = sim.people.date_cin2[cancer_genotypes, cancer_inds]
                date_cin3 = sim.people.date_cin3[cancer_genotypes, cancer_inds]
                hpv_time = (date_cin1 - date_infectious) * sim['dt']
                cin1_time = (date_cin2 - date_cin1) * sim['dt']
                cin2_time = (date_cin3 - date_cin2) * sim['dt']
                cin3_time = (sim.t - date_cin3) * sim['dt']
                total_time = (sim.t - date_exposed) * sim['dt']
                latent_inds = hpv.defined(date_latent)
                not_latent_inds = hpv.undefined(date_latent)
                self.dwelltime['precin']['Latent'] += hpv_time[latent_inds].tolist()
                self.dwelltime['cin1']['Latent'] += cin1_time[latent_inds].tolist()
                self.dwelltime['cin2']['Latent'] += cin2_time[latent_inds].tolist()
                self.dwelltime['cin3']['Latent'] += cin3_time[latent_inds].tolist()
                self.dwelltime['total']['Latent'] += total_time[latent_inds].tolist()
                self.age_causal['Latent'] += (current_age[latent_inds] - total_time[latent_inds]).tolist()
                self.age_cancer['Latent'] += (current_age[latent_inds]).tolist()

                self.dwelltime['precin']['None'] += hpv_time[not_latent_inds].tolist()
                self.dwelltime['cin1']['None'] += cin1_time[not_latent_inds].tolist()
                self.dwelltime['cin2']['None'] += cin2_time[not_latent_inds].tolist()
                self.dwelltime['cin3']['None'] += cin3_time[not_latent_inds].tolist()
                self.dwelltime['total']['None'] += total_time[not_latent_inds].tolist()
                self.age_causal['None'] += (current_age[not_latent_inds] - total_time[not_latent_inds]).tolist()
                self.age_cancer['None'] += (current_age[not_latent_inds]).tolist()

        return


class econ_analyzer(hpv.Analyzer):
    '''
    Analyzer for feeding into costing/health economic analysis.
    
    Produces a dataframe by year storing:
        
        - Resource use: number of vaccines, screens, lesions treated, cancers treated
        - Cases/deaths: number of new cancer cases and cancer deaths
        - Average age of new cases, average age of deaths, average age of noncancer death
    '''

    def __init__(self, start=2020, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start=start
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


    def apply(self, sim):
        if sim.yearvec[sim.t]>=self.start:
            ppl = sim.people
            def av_age(arr):
                if len(hpv.true(arr)): return np.mean(sim.people.age[hpv.true(arr)])
                else: return np.nan
            li = np.floor(sim.yearvec[sim.t])
            ltt = int((sim.t-1)*sim['dt']) # this is the timestep number vs year of sim, needed to retrieve outcomes from interventions
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
            self.df.loc[li].av_age_other_deaths = av_age(ppl.date_dead_other == lt)
            self.df.loc[li].av_age_cancer_deaths = av_age(ppl.date_dead_cancer == lt)
            self.df.loc[li].av_age_cancers = av_age(ppl.date_cancerous == lt)
        return


    def finalize(self, sim):
        # Add in results that are already generated (NB, these have all been scaled already)
        self.df['new_cancers'] = sim.results['cancers'][self.si:]
        self.df['new_cancer_deaths'] = sim.results['cancer_deaths'][self.si:]
        self.df['new_other_deaths'] = sim.results['other_deaths'][self.si:]
        return

class outcomes_by_year(hpv.Analyzer):
    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.interval = 1
        self.durations = np.arange(0, 41, self.interval)
        result_keys = ['cleared', 'persisted', 'progressed', 'cancer', 'dead', 'dead_cancer', 'dead_other', 'total']
        self.results = {rkey: np.zeros_like(self.durations) for rkey in result_keys}

    def initialize(self, sim):
        super().initialize(sim)
        if self.start_year is None:
            self.start_year = sim['start']

    def apply(self, sim):
        if sim.yearvec[sim.t] == self.start_year:
            idx = ((sim.people.date_exposed == sim.t) & (sim.people.sex==0) & (sim.people.n_infections==1)).nonzero()  # Get people exposed on this step
            inf_inds = idx[-1]
            if len(inf_inds):
                scale = sim.people.scale[inf_inds]
                time_to_clear = (sim.people.date_clearance[idx] - sim.t)*sim['dt']
                time_to_cancer = (sim.people.date_cancerous[idx] - sim.t)*sim['dt']
                time_to_cin = (sim.people.date_cin[idx] - sim.t)*sim['dt']

                # Count deaths. Note that there might be more people with a defined
                # cancer death date than with a defined cancer date because this is
                # counting all death, not just deaths resulting from infections on this
                # time step.
                time_to_cancer_death = (sim.people.date_dead_cancer[inf_inds] - sim.t)*sim['dt']
                time_to_other_death = (sim.people.date_dead_other[inf_inds] - sim.t)*sim['dt']

                for idd, dd in enumerate(self.durations):

                    dead_cancer = (time_to_cancer_death <= (dd+self.interval)) & ~(time_to_other_death <= (dd + self.interval))
                    dead_other = ~(time_to_cancer_death <= (dd + self.interval)) & (time_to_other_death <= (dd + self.interval))
                    dead = (time_to_cancer_death <= (dd + self.interval)) | (time_to_other_death <= (dd + self.interval))
                    cleared = ~dead & (time_to_clear <= (dd+self.interval))
                    persisted = ~dead & ~cleared & ~(time_to_cin <= (dd+self.interval)) # Haven't yet cleared or progressed
                    progressed = ~dead & ~cleared & (time_to_cin <= (dd+self.interval)) & ((time_to_clear>(dd+self.interval)) | (time_to_cancer > (dd+self.interval)))  # USing the ~ means that we also count nans
                    cancer = ~dead & (time_to_cancer <= (dd+self.interval))

                    dead_cancer_inds = hpv.true(dead_cancer)
                    dead_other_inds = hpv.true(dead_other)
                    dead_inds = hpv.true(dead)
                    cleared_inds = hpv.true(cleared)
                    persisted_inds = hpv.true(persisted)
                    progressed_inds = hpv.true(progressed)
                    cancer_inds = hpv.true(cancer)
                    derived_total = len(cleared_inds) + len(persisted_inds) + len(progressed_inds) + len(cancer_inds) + len(dead_inds)

                    if derived_total != len(inf_inds):
                        errormsg = "Something is wrong!"
                        raise ValueError(errormsg)
                    scaled_total = scale.sum()
                    self.results['cleared'][idd] += scale[cleared_inds].sum()#len(hpv.true(cleared))
                    self.results['persisted'][idd] += scale[persisted_inds].sum()#len(hpv.true(persisted_no_progression))
                    self.results['progressed'][idd] += scale[progressed_inds].sum()#len(hpv.true(persisted_with_progression))
                    self.results['cancer'][idd] += scale[cancer_inds].sum()#len(hpv.true(cancer))
                    self.results['dead'][idd] += scale[dead_inds].sum()
                    self.results['dead_cancer'][idd] += scale[dead_cancer_inds].sum()#len(hpv.true(dead))
                    self.results['dead_other'][idd] += scale[dead_other_inds].sum()  # len(hpv.true(dead))
                    self.results['total'][idd] += scaled_total#derived_total