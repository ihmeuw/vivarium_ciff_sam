from datetime import datetime
from typing import NamedTuple, Tuple

from scipy import stats

##################
# Time Constants #
##################

YEAR_DURATION: float = 365.25

##########################
# Cause Model Parameters #
##########################

# diarrhea duration in days
DIARRHEA_DURATION: int = 10

# measles duration in days
MEASLES_DURATION: int = 10

# LRI duration in days
LRI_DURATION: int = 10

# duration > bin_duration, so there is effectively no remission,
# and duration within the bin is bin_duration / 2
EARLY_NEONATAL_CAUSE_DURATION: float = 3.5


############################
# Wasting Model Parameters #
############################
class __Wasting(NamedTuple):
    # Wasting treatment coverage
    COVERAGE_START_AGE: float = 28 / YEAR_DURATION  # ~0.0767
    SAM_TX_COVERAGE: Tuple = ('sam_tx_coverage', stats.norm(loc=0.488, scale=0.0587))  # (0.604 - 0.374) / (2 * 1.96)
    MAM_TX_COVERAGE: Tuple = ('mam_tx_coverage', stats.norm(loc=0.488, scale=0.0587))  # (0.604 - 0.374) / (2 * 1.96)

    # Wasting treatment efficacy
    SAM_TX_EFFICACY: Tuple = ('sam_tx_efficacy', stats.norm(loc=0.700, scale=0.0306))  # (0.760 - 0.640) / (2 * 1.96)
    MAM_TX_EFFICACY: Tuple = ('mam_tx_efficacy', stats.norm(loc=0.731, scale=0.0745))  # (0.877 - 0.585) / (2 * 1.96)

    # Untreated time to recovery in days
    SAM_UX_RECOVERY_TIME: float = 62.0
    MAM_UX_RECOVERY_TIME: float = 63.0
    MILD_WASTING_UX_RECOVERY_TIME: float = 1000.0

    # Treated time to recovery in days
    SAM_TX_RECOVERY_TIME_OVER_6MO: float = 48.3
    SAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3
    MAM_TX_RECOVERY_TIME_OVER_6MO: float = 41.3
    MAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3


WASTING = __Wasting()


######################################
# Treatment and Prevention Constants #
######################################
class __SQLNS(NamedTuple):
    COVERAGE_START_AGE: float = 0.5
    COVERAGE_BASELINE: float = 0.0
    COVERAGE_RAMP_UP: float = 0.9
    EFFICACY_WASTING: float = 0.18
    EFFICACY_STUNTING_SEVERE: float = 0.15
    EFFICACY_STUNTING_MODERATE: float = 0.07


SQ_LNS = __SQLNS()


###################################
# Scale-up Intervention Constants #
###################################

SCALE_UP_START_DT = datetime(2023, 1, 1)
