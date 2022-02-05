from typing import NamedTuple

import pandas as pd

####################
# Project metadata #
####################

PROJECT_NAME = 'vivarium_ciff_sam'
CLUSTER_PROJECT = 'proj_cost_effect'

CLUSTER_QUEUE = 'all.q'
MAKE_ARTIFACT_MEM = '10G'
MAKE_ARTIFACT_CPU = '1'
MAKE_ARTIFACT_RUNTIME = '3:00:00'
MAKE_ARTIFACT_SLEEP = 10

YEAR_DURATION: float = 365.25

LOCATIONS = [
    'Ethiopia'
]

ARTIFACT_INDEX_COLUMNS = [
    'sex',
    'age_start',
    'age_end',
    'year_start',
    'year_end',
]

ARTIFACT_COLUMNS = pd.Index([f'draw_{i}' for i in range(0, 1000)])

GBD_2019_ROUND_ID = 6
GBD_2020_ROUND_ID = 7


class __AgeGroup(NamedTuple):
    BIRTH_ID = 164
    EARLY_NEONATAL_ID = 2
    LATE_NEONATAL_ID = 3
    MONTHS_1_TO_5 = 388
    MONTHS_6_TO_11 = 389
    MONTHS_12_TO_23 = 238
    YEARS_2_TO_4 = 34

    GBD_2019_LBWSG_EXPOSURE = {BIRTH_ID, EARLY_NEONATAL_ID, LATE_NEONATAL_ID}
    GBD_2019_LBWSG_RELATIVE_RISK = {EARLY_NEONATAL_ID, LATE_NEONATAL_ID}
    GBD_2019_SIDS = {LATE_NEONATAL_ID}
    GBD_2020 = {
        EARLY_NEONATAL_ID,
        LATE_NEONATAL_ID,
        MONTHS_1_TO_5,
        MONTHS_6_TO_11,
        MONTHS_12_TO_23,
        YEARS_2_TO_4,
    }


AGE_GROUP = __AgeGroup()
