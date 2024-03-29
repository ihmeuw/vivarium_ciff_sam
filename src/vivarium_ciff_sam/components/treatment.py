"""Prevention and treatment models"""
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.risks import Risk

from vivarium_ciff_sam.constants import data_keys, data_values, models
from vivarium_ciff_sam.utilities import get_random_variable


class SQLNSTreatment:
    """Manages SQ-LNS prevention"""

    def __init__(self):
        self.name = 'sq_lns'
        self._randomness_stream_name = f'initial_{self.name}_propensity'
        self.propensity_column_name = data_keys.SQ_LNS.PROPENSITY_COLUMN
        self.propensity_pipeline_name = data_keys.SQ_LNS.PROPENSITY_PIPELINE
        self.coverage_pipeline_name = data_keys.SQ_LNS.COVERAGE_PIPELINE

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        draw = builder.configuration.input_data.input_draw_number
        self.randomness = builder.randomness.get_stream(self._randomness_stream_name)

        self.wasting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_WASTING)
        self.severe_stunting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_STUNTING_SEVERE)
        self.moderate_stunting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_STUNTING_MODERATE)

        required_columns = [
            'age',
            self.propensity_column_name,
        ]

        self.propensity = builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: self.population_view.get(index)[self.propensity_column_name],
            requires_columns=[self.propensity_column_name]
        )

        self.coverage = builder.value.register_value_producer(
            self.coverage_pipeline_name,
            source=self.get_current_coverage,
            requires_columns=['age'],
            requires_values=[self.propensity_pipeline_name],
        )

        builder.value.register_value_modifier(
            f'{models.WASTING.MILD_STATE_NAME}_to_{models.WASTING.MODERATE_STATE_NAME}.transition_rate',
            modifier=self.apply_wasting_treatment,
            requires_values=[self.coverage_pipeline_name]
        )

        builder.value.register_value_modifier(
            'risk_factor.child_stunting.exposure_parameters',
            modifier=self.apply_stunting_treatment,
            requires_values=[self.coverage_pipeline_name]
        )

        self.population_view = builder.population.get_view(required_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.propensity_column_name],
                                                 requires_streams=[self._randomness_stream_name])

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(pd.Series(self.randomness.get_draw(pop_data.index),
                                              name=self.propensity_column_name))

    def get_current_coverage(self, index: pd.Index) -> pd.Series:
        age = self.population_view.get(index)['age']
        propensity = self.propensity(index)

        coverage = ((propensity < data_values.SQ_LNS.COVERAGE_BASELINE)
                    & (data_values.SQ_LNS.COVERAGE_START_AGE <= age))

        return coverage

    def apply_wasting_treatment(self, index: pd.Index, target: pd.Series) -> pd.Series:
        covered = self.coverage(index)
        target[covered] = target[covered] * self.wasting_risk_ratio

        return target

    def apply_stunting_treatment(self, index: pd.Index, target: pd.DataFrame) -> pd.Series:
        cat1_decrease = target.loc[:, 'cat1'] * (1 - self.severe_stunting_risk_ratio)
        cat2_decrease = target.loc[:, 'cat2'] * (1 - self.moderate_stunting_risk_ratio)

        covered = self.coverage(index)
        target.loc[covered, 'cat1'] = target.loc[covered, 'cat1'] - cat1_decrease.loc[covered]
        target.loc[covered, 'cat2'] = target.loc[covered, 'cat2'] - cat2_decrease.loc[covered]
        target.loc[covered, 'cat3'] = (target.loc[covered, 'cat3']
                                       + cat1_decrease.loc[covered] + cat2_decrease.loc[covered])
        return target


class WastingTreatment(Risk):

    def __init__(self, treatment_type: str):
        super().__init__(treatment_type)

        self.previous_wasting_column = f'previous_{data_keys.WASTING.name}'
        self.wasting_column = data_keys.WASTING.name

        self.treated_state = self._get_treated_state()

    ##########################
    # Initialization methods #
    ##########################

    def _get_treated_state(self) -> str:
        return self.risk.name.split('_treatment')[0]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        super().setup(builder)
        self._register_on_time_step_prepare_listener(builder)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name,
                                            self.previous_wasting_column,
                                            self.wasting_column])

    def _register_on_time_step_prepare_listener(self, builder: Builder) -> None:
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)

    ########################
    # Event-driven methods #
    ########################

    def on_time_step_cleanup(self, event: Event):
        pop = self.population_view.get(event.index)
        propensity = pop[self.propensity_column_name]
        remitted_mask = ((pop[self.previous_wasting_column] == self.treated_state)
                         & pop[self.wasting_column] != self.treated_state)
        propensity.loc[remitted_mask] = self.randomness.get_draw(remitted_mask.index)
        self.population_view.update(propensity)
