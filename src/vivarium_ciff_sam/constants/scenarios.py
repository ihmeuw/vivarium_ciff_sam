#############
# Scenarios #
#############

class Scenario:

    def __init__(self, has_alternative_treatment: bool, has_sqlns: bool):
        self.has_alternative_treatment = has_alternative_treatment
        self.has_sqlns = has_sqlns


SCENARIOS = {
    'baseline': Scenario(False, False),
    'wasting_treatment': Scenario(True, False),
    'sqlns': Scenario(True, True),
}
