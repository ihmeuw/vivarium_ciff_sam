from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import get_exposure_post_processor


class XFactorExposure(Risk):

    def __init__(self):
        super().__init__('risk_factor.x_factor')
        self.propensity_column_name = 'initial_child_wasting_propensity'
        self.propensity_pipeline_name = 'x_factor.propensity'
        self.exposure_pipeline_name = 'x_factor.exposure'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.propensity = self.get_propensity_pipeline(builder)
        self.exposure = self.get_exposure_pipeline(builder)
        self.population_view = self.get_population_view(builder)

    def get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: self.population_view.get(index)[self.propensity_column_name],
            requires_columns=[self.propensity_column_name]
        )

    def get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.get_current_exposure,
            requires_columns=['age', 'sex'],
            requires_values=[f'{self.risk.name}.propensity'],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

    def get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name])
