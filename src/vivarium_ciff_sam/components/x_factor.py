import pandas as pd

from vivarium import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks.effect import RiskEffect

from vivarium_ciff_sam.constants import data_keys, metadata, models


class XFactorExposure:

    configuration_defaults = {
        "x_factor": {
            "exposure": 'derived',
            "rebinned_exposed": [],
            "category_thresholds": [],
        },
        'effect_of_wasting_on_x_factor': {
            'exposure_parameters': {
                'susceptible_to_child_wasting': None,
                'mild_child_wasting': None,
                'moderate_acute_malnutrition': None,
                'severe_acute_malnutrition': None,
            }
        }
    }

    def __init__(self):
        self.name = 'x_factor_exposure'
        self.propensity_randomness_stream_name = 'x_factor_propensity_stream'
        self.propensity_column_name = 'x_factor_propensity'
        self.initial_wasting_column_name = 'initial_child_wasting'
        self.propensity_pipeline_name = 'x_factor.propensity'
        self.exposure_parameters_pipeline_name = 'risk_factor.x_factor.exposure_parameters'
        self.exposure_parameters_paf_pipeline_name = 'risk_factor.x_factor.exposure_parameters.paf'
        self.exposure_pipeline_name = 'x_factor.exposure'
        self.configuration_defaults = XFactorExposure.configuration_defaults

    def __repr__(self):
        return 'XFactorExposure'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = self.get_randomness_stream(builder)
        self.propensity = self.get_propensity_pipeline(builder)
        self.effect_of_wasting_on_x_factor = self.get_effect_of_wasting_on_x_factor(builder)

        self.exposure_parameters = self.get_exposure_parameters_pipeline(builder)
        self.exposure_parameters_paf = self.get_exposure_parameters_paf_pipeline(builder)
        self.exposure = self.get_exposure_pipeline(builder)

        self.population_view = self.get_population_view(builder)
        self.register_simulant_initializer(builder)

    def register_simulant_initializer(self, builder: Builder):
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.propensity_column_name],
            requires_streams=[self.propensity_randomness_stream_name]
        )

    # noinspection PyMethodMayBeStatic
    def get_randomness_stream(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(self.propensity_randomness_stream_name)

    def get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: self.population_view.get(index)[self.propensity_column_name],
            requires_columns=[self.propensity_column_name]
        )

    # noinspection PyMethodMayBeStatic
    def get_effect_of_wasting_on_x_factor(self, builder: Builder) -> ConfigTree:
        effects = builder.configuration['effect_of_wasting_on_x_factor']['exposure_parameters'].to_dict()
        return effects

    def get_exposure_parameters_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_parameters_pipeline_name,
            source=self.get_exposure_parameters,
            requires_columns=[self.initial_wasting_column_name]
        )

    def get_exposure_parameters_paf_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_parameters_paf_pipeline_name,
            source=lambda index: pd.Series(0.0, index=index)
        )

    def get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.get_exposure,
            requires_values=[self.propensity_pipeline_name, self.exposure_parameters_pipeline_name]
        )

    def get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.initial_wasting_column_name, self.propensity_column_name])

    def on_initialize_simulants(self, pop_data):
        propensities = self.randomness.get_draw(pop_data.index)
        propensities.name = self.propensity_column_name
        self.population_view.update(propensities)

    def get_exposure_parameters(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.subview([self.initial_wasting_column_name]).get(index)
        exposure_parameters = pd.Series(0, index=pop.index)
        exposure_parameters.name = self.exposure_parameters_pipeline_name
        for state, value in self.effect_of_wasting_on_x_factor.items():
            exposure_parameters[pop[self.initial_wasting_column_name] == state] = value
        return exposure_parameters

    def get_exposure(self, index: pd.Index) -> pd.Series:
        propensity = self.propensity(index)
        exposure_parameters = self.exposure_parameters(index)

        exposure = pd.Series('cat2', index=propensity.index)
        exposure.name = self.exposure_pipeline_name
        exposure[propensity < exposure_parameters] = 'cat1'
        return exposure


class XFactorEffect(RiskEffect):

    def __init__(self, target: str):
        super().__init__('risk_factor.x_factor', target)

    def load_population_attributable_fraction_data(self, builder: Builder):
        exposure_data = self._calculate_exposure(builder)
        relative_risk_data = self._calculate_rr(builder)

        mean_rr = (exposure_data * relative_risk_data).groupby(metadata.ARTIFACT_INDEX_COLUMNS).sum()
        paf_data = ((mean_rr - 1)/mean_rr).reset_index().rename(columns={0: 'value'})
        return paf_data

    @staticmethod
    def _calculate_exposure(builder: Builder) -> pd.Series:
        exposure_configs = builder.configuration.effect_of_child_wasting_on_x_factor.exposure_parameters.to_dict()

        wasting_exposure = (
            builder.data.load(data_keys.WASTING.EXPOSURE)
            .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ['parameter'])
        )
        exposure_parameters = pd.Series(
            {models.get_risk_category(k): v for k, v in exposure_configs.items()},
            index=wasting_exposure.index.get_level_values('parameter').unique()
        )

        cat1_exposure = (
            (wasting_exposure['value'] * exposure_parameters)
            .groupby(metadata.ARTIFACT_INDEX_COLUMNS)
            .sum()
            .reset_index()
        )
        cat1_exposure['parameter'] = 'cat1'

        cat2_exposure = cat1_exposure.copy()
        cat2_exposure.loc[:, 0] = 1 - cat1_exposure.loc[:, 0]
        cat2_exposure['parameter'] = 'cat2'
        return (
            pd.concat([cat1_exposure, cat2_exposure], ignore_index=True)
            .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ['parameter'])
            .loc[:, 0]
        )

    def _calculate_rr(self, builder: Builder) -> pd.Series:
        rr_value = (
            builder.configuration[f'effect_of_x_factor_on_{self.target.name}'][self.target.measure]['relative_risk']
        )

        cat1 = builder.data.load('population.demographic_dimensions')
        cat1['parameter'] = 'cat1'
        cat1['value'] = rr_value
        cat2 = cat1.copy()
        cat2['parameter'] = 'cat2'
        cat2['value'] = 1
        return (
            pd.concat([cat1, cat2], ignore_index=True)
            .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ['parameter'])
            .value
        )
