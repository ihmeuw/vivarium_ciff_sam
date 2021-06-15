##########################
# Cause Model Parameters #
##########################

# LRI duration in days
LRI_DURATION: int = 10
# lri_duration > bin_duration, so there is effectively no remission, and duration within the bin is bin_duration / 2
EARLY_NEONATAL_LRI_DURATION: float = 3.5


############################
# Wasting Model Parameters #
############################

# SAM (cat_1 wasting) duration in days
SAM_DURATION: int = 40

# MAM (cat_2 wasting) duration in days
MAM_DURATION: int = 70

# Mild wasting (cat_3 wasting) duration in days
MILD_WASTING_DURATION: int = 365


###################################
# Scale-up Intervention Constants #
###################################
