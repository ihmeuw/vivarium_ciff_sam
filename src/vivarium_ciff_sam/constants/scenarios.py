from typing import NamedTuple

#############
# Scenarios #
#############


class Scenario:

    def __init__(self, name: str, has_alternative_treatment: bool, has_sqlns: bool):
        self.name = name
        self.has_alternative_wasting_treatment = has_alternative_treatment
        self.has_sqlns = has_sqlns


class __Scenarios(NamedTuple):
    BASELINE: Scenario = Scenario('baseline', False, False)
    WASTING_TREATMENT: Scenario = Scenario('wasting_treatment', True, False)
    SQLNS: Scenario = Scenario('sqlns', True, True)

    def __getitem__(self, item) -> Scenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


SCENARIOS = __Scenarios()
