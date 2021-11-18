"""Prevention and treatment models"""
from typing import Dict, List

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline

from vivarium_public_health.utilities import EntityString
from vivarium_public_health.risks.distributions import SimulationDistribution
from vivarium_public_health.risks.data_transformations import get_exposure_post_processor

from vivarium_ciff_sam.components import DiseaseObserver
from vivarium_ciff_sam.constants import data_keys, data_values, models
from vivarium_ciff_sam.utilities import get_random_variable


class SQLNSTreatment:
    """Manages SQ-LNS prevention"""

    @property
    def name(self) -> str:
        """The name of this component."""
        return 'prevention_algorithm'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        draw = builder.configuration.input_data.input_draw_number
        self.randomness = builder.randomness.get_stream('initial_sq_lns_propensity')

        self.wasting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_WASTING)
        self.severe_stunting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_STUNTING_SEVERE)
        self.moderate_stunting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_STUNTING_MODERATE)

        propensity_col = 'sq_lns_propensity'
        required_columns = [
            'age',
            propensity_col,
        ]

        self.propensity = builder.value.register_value_producer(
            data_keys.SQ_LNS.PROPENSITY,
            source=lambda index: self.population_view.get(index)[propensity_col],
            requires_columns=[propensity_col]
        )

        self.coverage = builder.value.register_value_producer(
            data_keys.SQ_LNS.COVERAGE,
            source=self.get_current_coverage,
            requires_columns=['age'],
            requires_values=[data_keys.SQ_LNS.PROPENSITY],
        )

        builder.value.register_value_modifier(
            f'{models.WASTING.MILD_STATE_NAME}_to_{models.WASTING.MODERATE_STATE_NAME}.transition_rate',
            modifier=self.apply_wasting_treatment,
            requires_values=[data_keys.SQ_LNS.COVERAGE]
        )

        builder.value.register_value_modifier(
            'risk_factor.child_stunting.exposure_parameters',
            modifier=self.apply_stunting_treatment,
            requires_values=[data_keys.SQ_LNS.COVERAGE]
        )

        self.population_view = builder.population.get_view(required_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=[propensity_col],
                                                 requires_streams=['initial_sq_lns_propensity'])

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(pd.Series(self.randomness.get_draw(pop_data.index), name='sq_lns_propensity'))

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


class Risk:
    """A model for a risk factor defined by either a continuous or a categorical
    value. For example,

    #. high systolic blood pressure as a risk where the SBP is not dichotomized
       into hypotension and normal but is treated as the actual SBP
       measurement.
    #. smoking as two categories: current smoker and non-smoker.

    This component can source data either from builder.data or from parameters
    supplied in the configuration. If data is derived from the configuration, it
    must be an integer or float expressing the desired exposure level or a
    covariate name that is intended to be used as a proxy. For example, for a
    risk named "risk", the configuration could look like this:

    .. code-block:: yaml

       configuration:
           risk:
               exposure: 1.0

    or

    .. code-block:: yaml

       configuration:
           risk:
               exposure: proxy_covariate

    For polytomous risks, you can also provide an optional 'rebinned_exposed'
    block in the configuration to indicate that the risk should be rebinned
    into a dichotomous risk. That block should contain a list of the categories
    that should be rebinned into a single exposed category in the resulting
    dichotomous risk. For example, for a risk named "risk" with categories
    cat1, cat2, cat3, and cat4 that you wished to rebin into a dichotomous risk
    with an exposed category containing cat1 and cat2 and an unexposed category
    containing cat3 and cat4, the configuration could look like this:

    .. code-block:: yaml

       configuration:
           risk:
              rebinned_exposed: ['cat1', 'cat2']

    For alternative risk factors, you must provide a 'category_thresholds'
    block in the in configuration to dictate the thresholds that should be
    used to bin the continuous distributions. Note that this is mutually
    exclusive with providing 'rebinned_exposed' categories. For a risk named
    "risk", the configuration could look like:

    .. code-block:: yaml

       configuration:
           risk:
               category_thresholds: [7, 8, 9]

    """

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
            the type and name of a risk, specified as "type.name". Type is singular.
        """
        self.risk = EntityString(risk)
        self.configuration_defaults = self.get_configuration_defaults()
        self.exposure_distribution = self.get_exposure_distribution()
        self._sub_components = [self.exposure_distribution]

        self._randomness_stream_name = f'initial_{self.risk.name}_propensity'
        self.propensity_column_name = Risk.get_propensity_column_name(self.risk)
        self.propensity_pipeline_name = Risk.get_propensity_pipeline_name(self.risk)
        self.exposure_pipeline_name = Risk.get_exposure_pipeline_name(self.risk)

    def __repr__(self) -> str:
        return f"Risk({self.risk})"

    ##########################
    # Initialization methods #
    ##########################

    def get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            f"{self.risk.name}": {
                "exposure": 'data',
                "rebinned_exposed": [],
                "category_thresholds": [],
            }
        }

    def get_exposure_distribution(self) -> SimulationDistribution:
        return SimulationDistribution(self.risk)

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f'risk.{self.risk}'

    @property
    def sub_components(self) -> List:
        return self._sub_components

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = self.get_randomness_stream(builder)
        self.propensity = self.get_propensity_pipeline(builder)
        self.exposure = self.get_exposure_pipeline(builder)
        self.population_view = self.get_population_view(builder)
        self.register_simulant_initializer(builder)

    def get_randomness_stream(self, builder) -> RandomnessStream:
        return builder.randomness.get_stream(self._randomness_stream_name)

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
            requires_values=[self.propensity_pipeline_name],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

    def get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name])

    def register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.propensity_column_name],
            requires_streams=[self._randomness_stream_name]
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(self.randomness.get_draw(pop_data.index))

    ####################
    # Pipeline sources #
    ####################

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        propensity = self.propensity(index)
        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)

    ################
    # Name Getters #
    ################

    @staticmethod
    def get_propensity_column_name(risk_name: EntityString) -> str:
        return f'{risk_name.name}_propensity'

    @staticmethod
    def get_propensity_pipeline_name(risk_name: EntityString) -> str:
        return f'{risk_name.name}.propensity'

    @staticmethod
    def get_exposure_pipeline_name(risk_name: EntityString) -> str:
        return f'{risk_name.name}.exposure'


class WastingTreatment(Risk):

    def __init__(self, treatment_type: str):
        super().__init__(f'risk_factor.{treatment_type}')

        self.previous_wasting_column = DiseaseObserver.get_previous_state_column_name(data_keys.WASTING.name)
        self.wasting_column = data_keys.WASTING.name

        self.treated_state = self.get_treated_state()
        self.remission_states = self.get_remission_states()

    ##########################
    # Initialization methods #
    ##########################

    def get_treated_state(self) -> str:
        return models.get_risk_category(self.risk.name.split('_treatment')[0])

    def get_remission_states(self) -> List[str]:
        return [transition.to_state for transition in models.WASTING.TRANSITIONS
                if transition.from_state == self.treated_state]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        super().setup(builder)
        self.register_on_time_step_prepare_listener(builder)

    def get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name,
                                            self.previous_wasting_column,
                                            self.wasting_column])

    def register_on_time_step_prepare_listener(self, builder: Builder) -> None:
        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare, priority=9)

    ########################
    # Event-driven methods #
    ########################

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index)
        propensity = pop[self.propensity_column_name]
        remitted_pop = pop[(pop[self.previous_wasting_column] == self.treated_state)
                           & pop[self.wasting_column].isin(self.remission_states)]
        propensity[remitted_pop] = self.randomness.get_draw(remitted_pop.index)
        self.population_view.update(propensity)
