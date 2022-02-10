from datetime import datetime
from typing import Dict, NamedTuple, Tuple

import pandas as pd
from scipy import stats

from vivarium_ciff_sam.utilities import (
    get_norm_from_quantiles,
    get_lognorm_from_quantiles,
    get_truncnorm_from_quantiles,
    get_truncnorm_from_sd
)

#######################
# Universal Constants #
#######################

DAY_DURATION: float = 24
YEAR_DURATION: float = 365.25

##########################
# Cause Model Parameters #
##########################

# diarrhea duration in days
DIARRHEA_DURATION: Tuple = (
    'diarrheal_diseases_duration', get_norm_from_quantiles(mean=4.3, lower=4.2, upper=4.4)
)

# measles duration in days
MEASLES_DURATION: int = 10

# LRI duration in days
LRI_DURATION: Tuple = (
    'lri_duration', get_norm_from_quantiles(mean=7.79, lower=6.2, upper=9.64)
)

# duration > bin_duration, so there is effectively no remission,
# and duration within the bin is bin_duration / 2
EARLY_NEONATAL_CAUSE_DURATION: float = 3.5


############################
# Wasting Model Parameters #
############################
class __Wasting(NamedTuple):
    # Wasting age start (in years)
    START_AGE: float = 0.5

    # Wasting treatment distribution type and categories
    DISTRIBUTION: str = 'ordered_polytomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'Untreated',
        'cat2': 'Baseline treatment',
        'cat3': 'Alternative scenario treatment',
    }

    # Wasting treatment coverage
    COVERAGE_START_AGE: float = 28 / YEAR_DURATION  # ~0.0767
    BASELINE_SAM_TX_COVERAGE: Tuple = ('sam_tx_coverage', get_norm_from_quantiles(mean=0.488, lower=0.374, upper=0.604))
    BASELINE_MAM_TX_COVERAGE: Tuple = ('sam_tx_coverage', get_norm_from_quantiles(mean=0.15, lower=0.1, upper=0.2))
    ALTERNATIVE_TX_COVERAGE: float = 0.7

    # Wasting treatment efficacy
    BASELINE_SAM_TX_EFFICACY: Tuple = ('sam_tx_efficacy', get_norm_from_quantiles(mean=0.700, lower=0.64, upper=0.76))
    BASELINE_MAM_TX_EFFICACY: Tuple = ('mam_tx_efficacy', get_norm_from_quantiles(mean=0.731, lower=0.585, upper=0.877))
    SAM_TX_ALTERNATIVE_EFFICACY: float = 0.75
    MAM_TX_ALTERNATIVE_EFFICACY: float = 0.75

    # Incidence correction factor (total exit rate)
    SAM_K: float = ('sam_incidence_correction', get_lognorm_from_quantiles(median=6.7, lower=5.3, upper=8.4))

    # Untreated time to recovery in days
    MAM_UX_RECOVERY_TIME: float = 63.0
    DEFAULT_MILD_WASTING_UX_RECOVERY_TIME: float = 1_000.0

    # Treated time to recovery in days
    SAM_TX_RECOVERY_TIME_OVER_6MO: float = 48.3
    SAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3
    MAM_TX_RECOVERY_TIME_OVER_6MO: float = 41.3
    MAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3


WASTING = __Wasting()


##########################
# LBWSG Model Parameters #
##########################
class __LBWSG(NamedTuple):

    TMREL_GESTATIONAL_AGE_INTERVAL: pd.Interval = pd.Interval(38.0, 42.0)
    TMREL_BIRTH_WEIGHT_INTERVAL: pd.Interval = pd.Interval(3500.0, 4500.0)


LBWSG = __LBWSG()


###########################
# Maternal BMI Parameters #
###########################
class __MaternalMalnutrition(NamedTuple):
    DISTRIBUTION: str = 'dichotomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'BMI < 18.5',
        'cat2': 'BMI >= 18.5',
    }

    EXPOSURE: Tuple = (
        'maternal_malnutrition_exposure',
        get_truncnorm_from_quantiles(mean=0.224, lower=0.217, upper=0.231)
    )

    EXPOSED_BIRTH_WEIGHT_SHIFT: Tuple = (
        'maternal_malnutrition_effect_on_birth_weight',
        get_norm_from_quantiles(mean=-138.46, lower=-174.68, upper=-102.25)
    )


MATERNAL_MALNUTRITION = __MaternalMalnutrition()


######################################
# Treatment and Prevention Constants #
######################################
class __SQLNS(NamedTuple):
    COVERAGE_START_AGE: float = 0.5
    COVERAGE_BASELINE: float = 0.0
    COVERAGE_RAMP_UP: float = 0.9
    RISK_RATIO_WASTING: Tuple = ('sq_lns_wasting_effect',
                                 get_lognorm_from_quantiles(median=0.82, lower=0.74, upper=0.91))
    RISK_RATIO_STUNTING_SEVERE: Tuple = ('sq_lns_severe_stunting_effect',
                                         get_lognorm_from_quantiles(median=0.85, lower=0.74, upper=0.98))
    RISK_RATIO_STUNTING_MODERATE: Tuple = ('sq_lns_moderate_stunting_effect',
                                           get_lognorm_from_quantiles(median=0.93, lower=0.88, upper=0.98))


SQ_LNS = __SQLNS()


class __MaternalSupplementation(NamedTuple):

    DISTRIBUTION: str = 'dichotomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'uncovered',
        'cat2': 'covered',
    }

    BASELINE_IFA_COVERAGE: Tuple[str, stats.truncnorm] = (
        'ifa_coverage', get_truncnorm_from_quantiles(mean=0.598, lower=0.583, upper=0.613)
    )
    IFA_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        'ifa_birth_weight_shift', get_norm_from_quantiles(mean=57.73, lower=7.66, upper=107.79)
    )

    BASELINE_MMN_COVERAGE: float = 0.0
    MMN_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        'mmn_birth_weight_shift', get_norm_from_quantiles(mean=45.16, lower=32.31, upper=58.02)
    )

    BASELINE_BEP_COVERAGE: float = 0.0
    BEP_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        'bep_birth_weight_shift', get_norm_from_quantiles(mean=66.96, lower=13.13, upper=120.78)
    )

    ALTERNATIVE_COVERAGE: float = 0.9


MATERNAL_SUPPLEMENTATION = __MaternalSupplementation()


class __InsecticideTreatedNets(NamedTuple):
    DISTRIBUTION: str = 'dichotomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'uncovered',
        'cat2': 'covered',
    }

    EXPOSURE: Tuple = (
        'insecticide_treated_nets_exposure',
        get_truncnorm_from_quantiles(
            mean=0.44,
            lower=0.44 - (1.96 * 0.021),
            upper=0.44 + (1.96 * 0.021)
        )
    )

    BIRTH_WEIGHT_SHIFT: Tuple = (
        'insecticide_treated_nets_effect_on_birth_weight',
        get_norm_from_quantiles(mean=33, lower=5, upper=62, quantiles=(0.05, 0.95))
    )

    PROP_MALARIOUS: float = 0.6
    ALTERNATIVE_COVERAGE: float = 0.9 * PROP_MALARIOUS


INSECTICIDE_TX_NETS = __InsecticideTreatedNets()


class __ZincSupplementation(NamedTuple):
    DISTRIBUTION: str = 'dichotomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'uncovered',
        'cat2': 'covered',
    }

    BASELINE_PREVENTATIVE_COVERAGE: float = 0.0
    BASELINE_THERAPEUTIC_COVERAGE: Tuple[str, stats.truncnorm] = (
        'baseline_therapeutic_zinc_coverage', get_truncnorm_from_sd(mean=0.499, sd=0.143)
    )

    PREVENTATIVE_TX_EFFICACY: Tuple[str, stats.truncnorm] = (
        'preventative_zinc_treatment_efficacy',
        get_lognorm_from_quantiles(median=0.89, lower=0.82, upper=0.97, quantiles=(0.05, 0.95))
    )

    DIARRHEA_DURATION_SHIFT_HOURS: Tuple[str, stats.norm] = (
        'diarrhea_duration_shift',
        get_norm_from_quantiles(mean=-11.46, lower=-19.72, upper=-3.19, quantiles=(0.05, 0.95))
    )


PREVENTATIVE_ZINC = __ZincSupplementation()
THERAPEUTIC_ZINC = __ZincSupplementation()

#######################################
# Breastfeeding Risk Factor Constants #
#######################################

DISCONTINUED_BREASTFEEDING_START_AGE = 0.5
DISCONTINUED_BREASTFEEDING_END_AGE = 2.0
NON_EXCLUSIVE_BREASTFEEDING_END_AGE = 0.5


###################################
# Scale-up Intervention Constants #
###################################

SCALE_UP_START_DT = datetime(2023, 1, 1)
SCALE_UP_END_DT = datetime(2026, 1, 1)
