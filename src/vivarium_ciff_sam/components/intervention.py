import pandas as pd

from vivarium.framework.engine import Builder

from vivarium_ciff_sam.constants import data_keys, data_values, scenarios


class SQLNSIntervention:

    def __init__(self):
        self.name = 'sqlns_intervention'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        self.scenario = scenarios.SCENARIOS[builder.configuration.intervention.scenario]
        self.clock = builder.time.clock()

        required_columns = [
            "age"
        ]

        builder.value.register_value_modifier(
            data_keys.SQ_LNS.COVERAGE,
            modifier=self.coverage_effect,
            requires_columns=['age'],
            requires_values=[data_keys.SQ_LNS.PROPENSITY]
        )

        self.sq_lns_coverage_propensity = builder.value.get_value(data_keys.SQ_LNS.PROPENSITY)
        self.population_view = builder.population.get_view(required_columns)

    # define a function to do the modification
    def coverage_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        effect = False

        # if this is the alternative scenario and the scale up has already started calculate effect
        if self.scenario.has_sqlns and data_values.SCALE_UP_START_DT <= self.clock():
            age = self.population_view.get(idx)['age']
            propensity = self.sq_lns_coverage_propensity(idx)
            effect = ((data_values.SQ_LNS.COVERAGE_START_AGE <= age)
                      & (propensity < data_values.SQ_LNS.COVERAGE_RAMP_UP))

        return target | effect


class WastingTreatmentIntervention:

    def __init__(self):
        self.name = 'wasting_treatment_intervention'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        self.scenario = scenarios.SCENARIOS[builder.configuration.intervention.scenario]
        self.clock = builder.time.clock()

        # NOTE: this operation is NOT commutative. This pipeline must not be modified anywhere else.
        builder.value.register_value_modifier(
            f'risk_factor.{data_keys.SAM_TREATMENT.name}.exposure_parameters',
            modifier=self.coverage_effect,
        )
        builder.value.register_value_modifier(
            f'risk_factor.{data_keys.MAM_TREATMENT.name}.exposure_parameters',
            modifier=self.coverage_effect,
        )

    # define a function to do the modification
    # noinspection PyUnusedLocal
    def coverage_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        # if this is the alternative scenario and the scale up has already started update coverage
        if self.scenario.has_alternative_treatment and data_values.SCALE_UP_START_DT <= self.clock():
            target['cat1'] = 1 - data_values.WASTING.ALTERNATIVE_TX_COVERAGE
            target['cat2'] = 0.0
            target['cat3'] = data_values.WASTING.ALTERNATIVE_TX_COVERAGE

        return target
