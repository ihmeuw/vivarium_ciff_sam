from typing import NamedTuple


#############
# Scenarios #
#############

class __Scenarios(NamedTuple):
    BASELINE: str = 'baseline'
    TREATMENT: str = 'treatment_only'
    BOTH: str = 'treatment_and_prevention'


SCENARIOS = __Scenarios()
