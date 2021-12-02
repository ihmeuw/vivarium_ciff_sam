from typing import Dict

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.randomness import RandomnessStream

from vivarium_public_health.risks import Risk, RiskEffect


class XFactorExposure(Risk):

    def __init__(self):
        super().__init__('risk_factor.x_factor')
        self.propensity_column_name = 'initial_child_wasting_propensity'

    #################
    # Setup methods #
    #################

    def get_randomness_stream(self, builder) -> RandomnessStream:
        return None

    def register_simulant_initializer(self, builder: Builder) -> None:
        pass


class XFactorEffect(RiskEffect):

    configuration_defaults = {
        'effect_of_risk_on_target': {
            'conditional_exposure': 0.0,
            'measure': {
                'relative_risk': None,
                'mean': None,
                'se': None,
                'log_mean': None,
                'log_se': None,
                'tau_squared': None
            }
        }
    }

    def __init__(self, target: str):
        super().__init__('risk_factor.x_factor', target)

    ##########################
    # Initialization methods #
    ##########################

    def get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            f'effect_of_{self.risk.name}_on_{self.target.name}': {
                'conditional_exposure': XFactorEffect.configuration_defaults['effect_of_risk_on_target']['exposure'],
                self.target.measure: XFactorEffect.configuration_defaults['effect_of_risk_on_target']['measure']
            }
        }

    #################
    # Setup methods #
    #################

    def get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        source_key = f'effect_of_{self.risk.name}_on_{self.target.name}'
        exposure = builder.configuration[source_key]['conditional_exposure']
        rr = builder.configuration[source_key][self.target.measure]['relative_risk']

        paf = exposure * (rr - 1) / (exposure * (rr - 1) + 1)
        return builder.lookup.build_table(paf, key_columns=['sex'], parameter_columns=['age', 'year'])
