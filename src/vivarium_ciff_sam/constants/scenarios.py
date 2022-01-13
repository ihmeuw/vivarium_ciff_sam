from typing import NamedTuple

#############
# Scenarios #
#############


class Scenario:

    def __init__(
            self,
            name: str,
            has_alternative_treatment: bool = False,
            has_sqlns: bool = False,
            has_lbwsg: bool = False
    ):
        self.name = name
        self.has_alternative_wasting_treatment = has_alternative_treatment
        self.has_sqlns = has_sqlns
        self.has_lbwsg = has_lbwsg


class __Scenarios(NamedTuple):
    BASELINE: Scenario = Scenario('baseline')
    WASTING_TREATMENT: Scenario = Scenario('wasting_treatment', has_alternative_treatment=True)
    SQLNS: Scenario = Scenario('sqlns', has_alternative_treatment=True, has_sqlns=True)
    LBWSG_INTERVENTIONS: Scenario = Scenario(
        'lbwsg_interventions', has_alternative_treatment=True, has_sqlns=True, has_lbwsg=True
    )

    def __getitem__(self, item) -> Scenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


SCENARIOS = __Scenarios()
