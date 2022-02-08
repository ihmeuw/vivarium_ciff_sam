from typing import NamedTuple

from vivarium_ciff_sam.constants.data_values import WASTING

#############
# Scenarios #
#############


class InterventionScenario:

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


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario('baseline')
    WASTING_TREATMENT: InterventionScenario = InterventionScenario('wasting_treatment', has_alternative_treatment=True)
    SQLNS: InterventionScenario = InterventionScenario('sqlns', has_alternative_treatment=True, has_sqlns=True)
    LBWSG_INTERVENTIONS: InterventionScenario = InterventionScenario(
        'lbwsg_interventions', has_alternative_treatment=True, has_sqlns=True, has_lbwsg=True
    )

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()


class SamKScenario:

    def __init__(self, name: str, has_alternative_sam_k: bool = False):
        self.name = name
        self.distribution = WASTING.ALTERNATIVE_SAM_K if has_alternative_sam_k else WASTING.SAM_K


class __SamKScenarios(NamedTuple):
    BASELINE: SamKScenario = SamKScenario('baseline')
    ALTERNATIVE: SamKScenario = SamKScenario('alternative', has_alternative_sam_k=True)

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


SAM_K_SCENARIOS = __SamKScenarios()
