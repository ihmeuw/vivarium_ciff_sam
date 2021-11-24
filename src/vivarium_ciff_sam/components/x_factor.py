from vivarium.framework.engine import Builder
from vivarium.framework.randomness import RandomnessStream

from vivarium_public_health.risks import Risk


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
