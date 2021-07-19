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


###################################
# Scale-up Intervention Constants #
###################################
