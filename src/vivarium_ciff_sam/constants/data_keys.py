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
    DURATION: TargetString = TargetString('cause.diarrheal_diseases.duration')
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
    DURATION: TargetString = TargetString('cause.lower_respiratory_infections.duration')
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


class __ProteinEnergyMalnutrition(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    MAM_DISABILITY_WEIGHT: TargetString = TargetString('sequela.moderate_acute_malnutrition.disability_weight')
    SAM_DISABILITY_WEIGHT: TargetString = TargetString('sequela.severe_acute_malnutrition.disability_weight')
    EMR: TargetString = TargetString('cause.protein_energy_malnutrition.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.protein_energy_malnutrition.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.protein_energy_malnutrition.restrictions')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'protein_energy_malnutrition'

    @property
    def log_name(self):
        return 'protein energy malnutrition'


PEM = __ProteinEnergyMalnutrition()


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
    CAT4 = 'cat4'
    CAT3 = 'cat3'
    CAT2 = 'cat2'
    CAT1 = 'cat1'

    @property
    def name(self):
        return 'child_wasting'

    @property
    def log_name(self):
        return 'child wasting'


WASTING = __Wasting()


class __Stunting(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = 'risk_factor.child_stunting.distribution'
    ALT_DISTRIBUTION: TargetString = 'alternative_risk_factor.child_stunting.distribution'
    CATEGORIES: TargetString = 'risk_factor.child_stunting.categories'
    EXPOSURE: TargetString = 'risk_factor.child_stunting.exposure'
    RELATIVE_RISK: TargetString = 'risk_factor.child_stunting.relative_risk'
    PAF: TargetString = 'risk_factor.child_stunting.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT4 = 'cat4'
    CAT3 = 'cat3'
    CAT2 = 'cat2'
    CAT1 = 'cat1'

    @property
    def name(self):
        return 'child_stunting'

    @property
    def log_name(self):
        return 'child stunting'


STUNTING = __Stunting()


class __SQLNS(NamedTuple):
    PROPENSITY_COLUMN = 'sq_lns_propensity'
    PROPENSITY_PIPELINE = 'sq_lns.propensity'
    COVERAGE_PIPELINE = 'sq_lns.coverage'

    @property
    def name(self):
        return 'sq_lns'

    @property
    def log_name(self):
        return 'sq-lns'


SQ_LNS = __SQLNS()


class __WastingTreatment(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString
    DISTRIBUTION: TargetString
    CATEGORIES: TargetString
    RELATIVE_RISK: TargetString
    PAF: TargetString

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    UNCOVERED = 'cat1'
    BASELINE_COVERAGE = 'cat2'
    ALTERNATIVE_COVERAGE = 'cat3'

    TMREL_CATEGORY = BASELINE_COVERAGE
    COVERED_CATEGORIES = [BASELINE_COVERAGE, ALTERNATIVE_COVERAGE]
    UNCOVERED_CATEGORIES = [UNCOVERED]

    @property
    def name(self):
        return self.EXPOSURE.name

    @property
    def log_name(self):
        return self.name.replace('_', ' ')


def _get_wasting_treatment_keys(treatment_type: str) -> __WastingTreatment:
    return __WastingTreatment(
        EXPOSURE=TargetString(f'risk_factor.{treatment_type}.exposure'),
        DISTRIBUTION=TargetString(f'risk_factor.{treatment_type}.distribution'),
        CATEGORIES=TargetString(f'risk_factor.{treatment_type}.categories'),
        RELATIVE_RISK=TargetString(f'risk_factor.{treatment_type}.relative_risk'),
        PAF=TargetString(f'risk_factor.{treatment_type}.population_attributable_fraction'),
    )


SAM_TREATMENT = _get_wasting_treatment_keys('severe_acute_malnutrition_treatment')
MAM_TREATMENT = _get_wasting_treatment_keys('moderate_acute_malnutrition_treatment')


class __LowBirthWeightShortGestation(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.exposure'
    DISTRIBUTION: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.distribution'
    CATEGORIES: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.categories'
    RELATIVE_RISK: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.relative_risk'
    RELATIVE_RISK_INTERPOLATOR: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.relative_risk_interpolator'

    PAF: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    BIRTH_WEIGHT_EXPOSURE = TargetString('risk_factor.low_birth_weight.birth_exposure')

    @property
    def name(self):
        return 'low_birth_weight_and_short_gestation'

    @property
    def log_name(self):
        return 'low birth weight and short gestation'


LBWSG = __LowBirthWeightShortGestation()


class NonExclusiveBreastfeeding(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = 'risk_factor.non_exclusive_breastfeeding.distribution'
    CATEGORIES: TargetString = 'risk_factor.non_exclusive_breastfeeding.categories'
    EXPOSURE: TargetString = 'risk_factor.non_exclusive_breastfeeding.exposure'
    RELATIVE_RISK: TargetString = 'risk_factor.non_exclusive_breastfeeding.relative_risk'
    PAF: TargetString = 'risk_factor.non_exclusive_breastfeeding.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT4 = 'cat4'
    CAT3 = 'cat3'
    CAT2 = 'cat2'
    CAT1 = 'cat1'

    @property
    def name(self):
        return 'non_exclusive_breastfeeding'

    @property
    def log_name(self):
        return 'non-exclusive breastfeeding'


NON_EXCLUSIVE_BREASTFEEDING = NonExclusiveBreastfeeding()


class DiscontinuedBreastfeeding(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = 'risk_factor.discontinued_breastfeeding.distribution'
    CATEGORIES: TargetString = 'risk_factor.discontinued_breastfeeding.categories'
    EXPOSURE: TargetString = 'risk_factor.discontinued_breastfeeding.exposure'
    RELATIVE_RISK: TargetString = 'risk_factor.discontinued_breastfeeding.relative_risk'
    PAF: TargetString = 'risk_factor.discontinued_breastfeeding.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT2 = 'cat2'
    CAT1 = 'cat1'

    @property
    def name(self):
        return 'discontinued_breastfeeding'

    @property
    def log_name(self):
        return 'discontinued breastfeeding'


DISCONTINUED_BREASTFEEDING = DiscontinuedBreastfeeding()


class PreventativeZinc(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = 'risk_factor.preventative_zinc.distribution'
    CATEGORIES: TargetString = 'risk_factor.preventative_zinc.categories'
    EXPOSURE: TargetString = 'risk_factor.preventative_zinc.exposure'
    RELATIVE_RISK: TargetString = 'risk_factor.preventative_zinc.relative_risk'
    PAF: TargetString = 'risk_factor.preventative_zinc.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT2 = 'cat2'
    CAT1 = 'cat1'

    AFFECTED_ENTITY = 'diarrheal_diseases'
    AFFECTED_MEASURE = 'incidence_rate'

    @property
    def name(self):
        return 'preventative_zinc'

    @property
    def log_name(self):
        return 'preventative zinc'


PREVENTATIVE_ZINC = PreventativeZinc()


class TherapeuticZinc(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = 'risk_factor.therapeutic_zinc.distribution'
    CATEGORIES: TargetString = 'risk_factor.therapeutic_zinc.categories'
    EXPOSURE: TargetString = 'risk_factor.therapeutic_zinc.exposure'
    RELATIVE_RISK: TargetString = 'risk_factor.therapeutic_zinc.relative_risk'
    PAF: TargetString = 'risk_factor.therapeutic_zinc.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT2 = 'cat2'
    CAT1 = 'cat1'

    AFFECTED_ENTITY = 'diarrheal_diseases'
    AFFECTED_MEASURE = 'remission_rate'

    @property
    def name(self):
        return 'therapeutic_zinc'

    @property
    def log_name(self):
        return 'therapeutic zinc'


THERAPEUTIC_ZINC = TherapeuticZinc()


class __AffectedUnmodeledCauses(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    URI_CSMR: TargetString = TargetString('cause.upper_respiratory_infections.cause_specific_mortality_rate')
    OTITIS_MEDIA_CSMR: TargetString = TargetString('cause.otitis_media.cause_specific_mortality_rate')
    MENINGITIS_CSMR: TargetString = TargetString('cause.meningitis.cause_specific_mortality_rate')
    ENCEPHALITIS_CSMR: TargetString = TargetString('cause.encephalitis.cause_specific_mortality_rate')
    NEONATAL_PRETERM_BIRTH_CSMR: TargetString = TargetString('cause.neonatal_preterm_birth.cause_specific_mortality_rate')
    NEONATAL_ENCEPHALOPATHY_CSMR: TargetString = TargetString('cause.neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.cause_specific_mortality_rate')
    NEONATAL_SEPSIS_CSMR: TargetString = TargetString('cause.neonatal_sepsis_and_other_neonatal_infections.cause_specific_mortality_rate')
    NEONATAL_JAUNDICE_CSMR: TargetString = TargetString('cause.hemolytic_disease_and_other_neonatal_jaundice.cause_specific_mortality_rate')
    OTHER_NEONATAL_DISORDERS_CSMR: TargetString = TargetString('cause.other_neonatal_disorders.cause_specific_mortality_rate')
    SIDS_CSMR: TargetString = TargetString('cause.sudden_infant_death_syndrome.cause_specific_mortality_rate')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'affected_unmodeled_causes'

    @property
    def log_name(self):
        return 'affected unmodeled causes'


AFFECTED_UNMODELED_CAUSES = __AffectedUnmodeledCauses()


class __AdditiveRisk(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString
    DISTRIBUTION: TargetString
    CATEGORIES: TargetString
    # analogous to excess mortality rate
    EXCESS_SHIFT: TargetString
    # analogous to cause specific mortality rate
    RISK_SPECIFIC_SHIFT: TargetString

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT1 = 'cat1'
    CAT2 = 'cat2'

    @property
    def name(self):
        return self.EXPOSURE.name

    @property
    def log_name(self):
        return self.name.replace('_', ' ')


def _get_additive_risk_keys(treatment_type: str) -> __AdditiveRisk:
    return __AdditiveRisk(
        EXPOSURE=TargetString(f'risk_factor.{treatment_type}.exposure'),
        DISTRIBUTION=TargetString(f'risk_factor.{treatment_type}.distribution'),
        CATEGORIES=TargetString(f'risk_factor.{treatment_type}.categories'),
        EXCESS_SHIFT=TargetString(f'risk_factor.{treatment_type}.excess_shift'),
        RISK_SPECIFIC_SHIFT=TargetString(f'risk_factor.{treatment_type}.risk_specific_shift'),
    )


MATERNAL_MALNUTRITION = _get_additive_risk_keys('maternal_malnutrition')
IFA_SUPPLEMENTATION = _get_additive_risk_keys('iron_folic_acid_supplementation')
MMN_SUPPLEMENTATION = _get_additive_risk_keys('multiple_micronutrient_supplementation')
BEP_SUPPLEMENTATION = _get_additive_risk_keys('balanced_energy_protein_supplementation')
INSECTICIDE_TX_NETS = _get_additive_risk_keys('insecticide_treated_nets')


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    DIARRHEA,
    MEASLES,
    LRI,
    PEM,
    WASTING,
    STUNTING,
    SQ_LNS,
    SAM_TREATMENT,
    MAM_TREATMENT,
    LBWSG,
    AFFECTED_UNMODELED_CAUSES,
    MATERNAL_MALNUTRITION,
    IFA_SUPPLEMENTATION,
    MMN_SUPPLEMENTATION,
    BEP_SUPPLEMENTATION,
    PREVENTATIVE_ZINC,
    THERAPEUTIC_ZINC,
    INSECTICIDE_TX_NETS,
    # NON_EXCLUSIVE_BREASTFEEDING,
    # DISCONTINUED_BREASTFEEDING,
]
