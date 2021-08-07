from datetime import datetime
from typing import NamedTuple

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

# Wasting duration in days
SAM_DURATION: int = 40
MAM_DURATION: int = 70
MILD_WASTING_DURATION: int = 365

# Wasting treatment coverage
SAM_TX_COVERAGE: float = 0.488
MAM_TX_COVERAGE: float = 0.488

# Untreated time to recovery in days
SAM_UX_RECOVERY_TIME: float = 60.5
MAM_UX_RECOVERY_TIME: float = 63.0
MILD_WASTING_UX_RECOVERY_TIME: float = 1000.0

# Treated time to recovery in days
SAM_TX_RECOVERY_TIME: float = 48.3
MAM_TX_RECOVERY_TIME: float = 41.3


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
