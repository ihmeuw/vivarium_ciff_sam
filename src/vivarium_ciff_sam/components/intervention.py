import pandas as pd

from vivarium.framework.engine import Builder

from vivarium_ciff_sam.constants import data_keys, data_values, scenarios


class Intervention:

    configuration_defaults = {
        'treatment': {
            'scenario': scenarios.SCENARIOS.BASELINE
        }
    }

    def __init__(self):
        self.name = 'intervention'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Perform this component's setup."""
        self.scenario = builder.configuration.treatment.scenario
        self.clock = builder.time.clock()

        required_columns = [
            "age"
        ]

        builder.value.register_value_modifier(
            data_keys.SQ_LNS.COVERAGE,
            modifier=self.sq_lns_coverage_intervention_effect,
            requires_columns=['age'],
            requires_values=[data_keys.SQ_LNS.PROPENSITY]
        )

        self.sq_lns_coverage_propensity = builder.value.get_value(data_keys.SQ_LNS.PROPENSITY)
        self.population_view = builder.population.get_view(required_columns)

    # define a function to do the modification
    def sq_lns_coverage_intervention_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        effect = False

        # if this is the alternative scenario and the scale up has already started calculate effect
        if self.scenario == scenarios.SCENARIOS.BOTH and data_values.SCALE_UP_START_DT <= self.clock():
            age = self.population_view.get(idx)['age']
            propensity = self.sq_lns_coverage_propensity(idx)
            effect = ((data_values.SQ_LNS.COVERAGE_START_AGE <= age)
                      & (propensity < data_values.SQ_LNS.COVERAGE_RAMP_UP))

        return target | effect
