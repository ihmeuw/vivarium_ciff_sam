"""Prevention and treatment models"""
import pandas as pd
from vivarium.framework.engine import Builder

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
        self.randomness = builder.randomness.get_stream('initial_sq_lns_propensity')

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
        target[covered] = target[covered] * (1 - data_values.SQ_LNS.EFFICACY_WASTING)

        return target

    def apply_stunting_treatment(self, index: pd.Index, target: pd.DataFrame) -> pd.Series:
        cat1_decrease = target.loc[:, 'cat1'] * data_values.SQ_LNS.EFFICACY_STUNTING_SEVERE
        cat2_decrease = target.loc[:, 'cat2'] * data_values.SQ_LNS.EFFICACY_STUNTING_MODERATE

        covered = self.coverage(index)
        target.loc[covered, 'cat1'] = target.loc[covered, 'cat1'] - cat1_decrease.loc[covered]
        target.loc[covered, 'cat2'] = target.loc[covered, 'cat2'] - cat2_decrease.loc[covered]
        target.loc[covered, 'cat3'] = (target.loc[covered, 'cat3']
                                       + cat1_decrease.loc[covered] + cat2_decrease.loc[covered])
        return target


class WastingTreatment:
    """Manages wasting treatment and maintenance measures"""

    @property
    def name(self) -> str:
        """The name of this component."""
        return 'treatment_algorithm'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.treatment_randomness = builder.randomness.get_stream('initial_wasting_treatment_propensity')
        self.efficacy_randomness = builder.randomness.get_stream('initial_wasting_treatment_efficacy_propensity')

        draw = builder.configuration.input_data.input_draw_number

        treatment_propensity_col = 'wasting_treatment_propensity'
        efficacy_propensity_col = 'wasting_treatment_efficacy_propensity'
        required_columns = [
            'age',
            treatment_propensity_col,
            efficacy_propensity_col,
        ]

        # Coverage levels for SAM and MAM treatment
        self.sam_treatment_coverage_level = get_random_variable(draw, *data_values.WASTING.SAM_TX_COVERAGE)
        self.mam_treatment_coverage_level = get_random_variable(draw, *data_values.WASTING.MAM_TX_COVERAGE)

        # Treatment efficacy for SAM and MAM
        self.sam_treatment_efficacy_level = get_random_variable(draw, *data_values.WASTING.SAM_TX_EFFICACY)
        self.mam_treatment_efficacy_level = get_random_variable(draw, *data_values.WASTING.MAM_TX_EFFICACY)

        # Individuals propensities for treatment and treatment efficacy. Same values used for both SAM and MAM
        self.treatment_propensity = builder.value.register_value_producer(
            data_keys.WASTING_TREATMENT.PROPENSITY,
            source=lambda index: self.population_view.get(index)[treatment_propensity_col],
            requires_columns=[treatment_propensity_col]
        )

        self.efficacy_propensity = builder.value.register_value_producer(
            data_keys.WASTING_TREATMENT.EFFICACY_PROPENSITY,
            source=lambda index: self.population_view.get(index)[efficacy_propensity_col],
            requires_columns=[efficacy_propensity_col]
        )

        # Pipeline for SAM and MAM treatment coverage state
        self.sam_treatment_coverage = builder.value.register_value_producer(
            data_keys.WASTING_TREATMENT.SAM_COVERAGE,
            source=self.get_sam_treatment_coverage_source,
            requires_columns=['age'],
            requires_values=[data_keys.WASTING_TREATMENT.PROPENSITY],
        )

        self.mam_treatment_coverage = builder.value.register_value_producer(
            data_keys.WASTING_TREATMENT.MAM_COVERAGE,
            source=self.get_mam_treatment_coverage_source,
            requires_columns=['age'],
            requires_values=[data_keys.WASTING_TREATMENT.PROPENSITY],
        )

        # Modifiers to apply treatment (or non-treatment) effects to SAM and MAM remission transitions
        builder.value.register_value_modifier(
            f'{models.WASTING.SEVERE_STATE_NAME}_to_{models.WASTING.MILD_STATE_NAME}.transition_rate',
            modifier=self.apply_sam_treatment_modifier,
            requires_columns=['age'],
            requires_values=[data_keys.WASTING_TREATMENT.EFFECTIVE_COVERAGE]
        )

        builder.value.register_value_modifier(
            f'{models.WASTING.SEVERE_STATE_NAME}_to_{models.WASTING.MODERATE_STATE_NAME}.transition_rate',
            modifier=self.apply_sam_non_treatment_modifier,
            requires_values=[data_keys.WASTING_TREATMENT.EFFECTIVE_COVERAGE]
        )

        builder.value.register_value_modifier(
            f'{models.WASTING.MODERATE_STATE_NAME}_to_{models.WASTING.MILD_STATE_NAME}.transition_rate',
            modifier=self.apply_mam_treatment_modifier,
            requires_values=[data_keys.WASTING_TREATMENT.EFFECTIVE_COVERAGE]
        )

        self.population_view = builder.population.get_view(required_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[treatment_propensity_col, efficacy_propensity_col],
                                                 requires_streams=['initial_wasting_treatment_propensity',
                                                                   'initial_wasting_treatment_efficacy_propensity'])

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(pd.Series(self.treatment_randomness.get_draw(pop_data.index),
                                              name='sq_lns_propensity'))

    def get_sam_treatment_coverage_source(self, index: pd.Index) -> pd.Series:
        coverage = self._treatment_coverage_source('sam', index)
        return coverage

    def get_mam_treatment_coverage_source(self, index: pd.Index) -> pd.Series:
        coverage = self._treatment_coverage_source('mam', index)
        return coverage

    def _treatment_coverage_source(self, treatment_type: str, index: pd.Index) -> pd.Series:
        treatment_parameters = {
            'sam': (self.sam_treatment_coverage_level, self.sam_treatment_efficacy_level),
            'mam': (self.mam_treatment_coverage_level, self.mam_treatment_efficacy_level),
        }
        treatment_coverage_level, treatment_efficacy_level = treatment_parameters[treatment_type]

        age = self.population_view.get(index)['age']
        treatment_propensity = self.treatment_propensity(index)
        efficacy_propensity = self.efficacy_propensity(index)

        ineligible = age < data_values.WASTING.COVERAGE_START_AGE
        untreated = ~ineligible & (treatment_coverage_level <= treatment_propensity)
        non_responsive = ~ineligible & ~untreated & (treatment_efficacy_level <= efficacy_propensity)
        effectively_covered = ~ineligible & ~untreated & (efficacy_propensity < treatment_efficacy_level)

        coverage = pd.Series(index=index)
        coverage[ineligible] = data_keys.WASTING_TREATMENT.INELIGIBLE
        coverage[untreated] = data_keys.WASTING_TREATMENT.UNTREATED
        coverage[non_responsive] = data_keys.WASTING_TREATMENT.NON_RESPONSIVE
        coverage[effectively_covered] = data_keys.WASTING_TREATMENT.EFFECTIVELY_COVERED

        return coverage

    def apply_sam_treatment_modifier(self, index: pd.Index, target: pd.Series) -> pd.Series:
        age = self.population_view.get(index)['age']
        coverage = self.get_sam_treatment_coverage_source(index)

        young_effective = (coverage == data_keys.WASTING_TREATMENT.EFFECTIVELY_COVERED) & (age < 0.5)
        old_effective = (coverage == data_keys.WASTING_TREATMENT.EFFECTIVELY_COVERED) & (0.5 <= age)
        unaffected = (coverage.isin([data_keys.WASTING_TREATMENT.UNTREATED,
                                     data_keys.WASTING_TREATMENT.NON_RESPONSIVE])
                      & (data_values.WASTING.COVERAGE_START_AGE < age))

        target[young_effective] = 1 / data_values.WASTING.SAM_TX_RECOVERY_TIME_UNDER_6MO
        target[old_effective] = 1 / data_values.WASTING.SAM_TX_RECOVERY_TIME_OVER_6MO
        target[unaffected] = 0
        return target

    def apply_sam_non_treatment_modifier(self, index: pd.Index, target: pd.Series) -> pd.Series:
        coverage = self.get_sam_treatment_coverage_source(index)

        effective = coverage == data_keys.WASTING_TREATMENT.EFFECTIVELY_COVERED
        unaffected = coverage.isin([data_keys.WASTING_TREATMENT.UNTREATED, data_keys.WASTING_TREATMENT.NON_RESPONSIVE])

        target[effective] = 0
        target[unaffected] = 1 / data_values.WASTING.SAM_UX_RECOVERY_TIME
        return target

    def apply_mam_treatment_modifier(self, index: pd.Index, target: pd.Series) -> pd.Series:
        age = self.population_view.get(index)['age']
        coverage = self.get_sam_treatment_coverage_source(index)

        young_effective = (coverage == data_keys.WASTING_TREATMENT.EFFECTIVELY_COVERED) & (age < 0.5)
        old_effective = (coverage == data_keys.WASTING_TREATMENT.EFFECTIVELY_COVERED) & (0.5 <= age)
        unaffected = (coverage.isin([data_keys.WASTING_TREATMENT.UNTREATED,
                                     data_keys.WASTING_TREATMENT.NON_RESPONSIVE])
                      & (data_values.WASTING.COVERAGE_START_AGE < age))

        target[young_effective] = 1 / data_values.WASTING.MAM_TX_RECOVERY_TIME_UNDER_6MO
        target[old_effective] = 1 / data_values.WASTING.MAM_TX_RECOVERY_TIME_OVER_6MO
        target[unaffected] = 1 / data_values.WASTING.MAM_UX_RECOVERY_TIME
        return target
