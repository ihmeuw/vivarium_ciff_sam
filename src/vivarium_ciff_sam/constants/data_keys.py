from typing import NamedTuple

from vivarium_public_health.utilities import TargetString


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    LOCATION: str = 'population.location'
    STRUCTURE: str = 'population.structure'
    AGE_BINS: str = 'population.age_bins'
    DEMOGRAPHY: str = 'population.demographic_dimensions'
    TMRLE: str = 'population.theoretical_minimum_risk_life_expectancy'
    ACMR: str = 'cause.all_causes.cause_specific_mortality_rate'
    CRUDE_BIRTH_RATE: str = 'covariate.live_births_by_sex.estimate'

    @property
    def name(self):
        return 'population'

    @property
    def log_name(self):
        return 'population'


POPULATION = __Population()


##########
# Causes #
##########


class __DiarrhealDiseases(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString('cause.diarrheal_diseases.prevalence')
    INCIDENCE_RATE: TargetString = TargetString('cause.diarrheal_diseases.incidence_rate')
    REMISSION_RATE: TargetString = TargetString('cause.diarrheal_diseases.remission_rate')
    DISABILITY_WEIGHT: TargetString = TargetString('cause.diarrheal_diseases.disability_weight')
    EMR: TargetString = TargetString('cause.diarrheal_diseases.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.diarrheal_diseases.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.diarrheal_diseases.restrictions')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'diarrheal_diseases'

    @property
    def log_name(self):
        return 'diarrheal diseases'


DIARRHEA = __DiarrhealDiseases()


class __Measles(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString('cause.measles.prevalence')
    INCIDENCE_RATE: TargetString = TargetString('cause.measles.incidence_rate')
    DISABILITY_WEIGHT: TargetString = TargetString('cause.measles.disability_weight')
    EMR: TargetString = TargetString('cause.measles.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.measles.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.measles.restrictions')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'measles'

    @property
    def log_name(self):
        return 'measles'


MEASLES = __Measles()


class __LowerRespiratoryInfections(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString('cause.lower_respiratory_infections.prevalence')
    INCIDENCE_RATE: TargetString = TargetString('cause.lower_respiratory_infections.incidence_rate')
    REMISSION_RATE: TargetString = TargetString('cause.lower_respiratory_infections.remission_rate')
    DISABILITY_WEIGHT: TargetString = TargetString('cause.lower_respiratory_infections.disability_weight')
    EMR: TargetString = TargetString('cause.lower_respiratory_infections.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.lower_respiratory_infections.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.lower_respiratory_infections.restrictions')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'lower_respiratory_infections'

    @property
    def log_name(self):
        return 'lower respiratory infections'


LRI = __LowerRespiratoryInfections()


################
# Risk Factors #
################


class __Wasting(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = 'risk_factor.child_wasting.distribution'
    ALT_DISTRIBUTION: TargetString = 'alternative_risk_factor.child_wasting.distribution'
    CATEGORIES: TargetString = 'risk_factor.child_wasting.categories'
    EXPOSURE: TargetString = 'risk_factor.child_wasting.exposure'
    RELATIVE_RISK: TargetString = 'risk_factor.child_wasting.relative_risk'
    PAF: TargetString = 'risk_factor.child_wasting.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    TMREL = 'cat4'
    MILD = 'cat3'
    MAM = 'cat2'
    SAM = 'cat1'

    @property
    def name(self):
        return 'child_wasting'

    @property
    def log_name(self):
        return 'child wasting'


WASTING = __Wasting()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    DIARRHEA,
    MEASLES,
    LRI,
    WASTING,
]
