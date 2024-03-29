from abc import abstractmethod, ABC
import pickle
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks import data_transformations
from vivarium_public_health.risks.distributions import SimulationDistribution

from vivarium_ciff_sam.constants import data_keys, data_values, metadata


class LBWSGSubRisk(Risk, ABC):
    """"
    Risk component for the individual aspects of LBWSG (i.e. birth weight and gestational age).
    `risk_factor.low_birth_weight_and_short_gestation` must exist.
    """

    RISK_NAME = 'low_birth_weight_and_short_gestation'

    def __init__(self, risk: str):
        super(LBWSGSubRisk, self).__init__(risk)
        self._sub_components = []
        self.birth_exposure_pipeline_name = f'{self.risk.name}.birth_exposure'
        self.lbwsg_exposure_pipeline_name = f'{data_keys.LBWSG.name}.exposure'
        self.exposure_column_name = f'{self.risk.name}_exposure'
        self.risk_specific_shift_pipeline_name = f'{self.risk.name}.risk_specific_shift'

    ##########################
    # Initialization methods #
    ##########################

    def _get_exposure_distribution(self) -> SimulationDistribution:
        return None

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.lbwsg_exposure = self._get_lbwsg_exposure_pipeline(builder)
        self.category_intervals = self._get_category_intervals(builder)
        self.risk_specific_shift = self._get_risk_specific_shift_pipeline(builder)
        self.birth_exposure_pipeline = self._get_birth_exposure_pipeline(builder)

    def _get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        # Propensity only used on initialization; not being saved to avoid a cycle
        return None

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self._get_current_exposure,
            requires_columns=[self.exposure_column_name],
            preferred_post_processor=data_transformations.get_exposure_post_processor(builder, self.risk)
        )

    def _get_birth_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.birth_exposure_pipeline_name,
            source=self._get_birth_exposure,
            requires_columns=['age', 'sex'],
            requires_values=[
                self.lbwsg_exposure_pipeline_name,
                self.risk_specific_shift_pipeline_name,
            ],
            requires_streams=[self._randomness_stream_name],
            preferred_post_processor=data_transformations.get_exposure_post_processor(builder, self.risk)
        )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(['tracked', self.exposure_column_name])

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.exposure_column_name],
            requires_streams=[self._randomness_stream_name],
            requires_values=[self.birth_exposure_pipeline_name]
        )

    def _get_lbwsg_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.get_value(self.lbwsg_exposure_pipeline_name)

    def _get_risk_specific_shift_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.risk_specific_shift_pipeline_name,
            source=builder.lookup.build_table(0),
        )

    @classmethod
    def _get_category_intervals(cls, builder: Builder) -> pd.Series:
        return cls.get_intervals_from_categories(builder.data.load(f'risk_factor.{data_keys.LBWSG.name}.categories'))

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        birth_exposure = pd.Series(self.birth_exposure_pipeline(pop_data.index), name=self.exposure_column_name)
        self.population_view.update(birth_exposure)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index: pd.Index) -> pd.Series:
        exposure = self.population_view.get(index)[self.exposure_column_name]
        return exposure

    def _get_birth_exposure(self, index: pd.Index) -> pd.Series:
        propensities = pd.Series(self.randomness.get_draw(index), name=self.propensity_column_name)
        lbwsg_categories = self.lbwsg_exposure(index)
        risk_specific_shift = self.risk_specific_shift(index)

        def get_exposure_from_category(row: pd.Series) -> float:
            category_interval = self.category_intervals[row[lbwsg_categories.name]]
            exposure = (
                row[propensities.name]
                * (category_interval.right - category_interval.left)
                + category_interval.left
            )
            return exposure

        risk_deleted_exposure = (
            pd.concat([lbwsg_categories, propensities], axis=1)
            .apply(get_exposure_from_category, axis=1)
            .sub(risk_specific_shift)
        )
        risk_deleted_exposure.name = self.exposure_pipeline_name
        return risk_deleted_exposure

    ##################
    # Helper methods #
    ##################

    @classmethod
    def get_intervals_from_categories(cls, categories: Dict[str, str]) -> pd.Series:
        category_endpoints = pd.Series(
            {cat: cls.parse_description(description) for cat, description in categories.items()},
            name=f'{cls.RISK_NAME}.endpoints'
        )
        category_endpoints.index.name = 'parameter'
        return category_endpoints

    @staticmethod
    @abstractmethod
    def parse_description(description: str) -> pd.Interval:
        # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'
        pass


class LowBirthWeight(LBWSGSubRisk):

    RISK_NAME = 'low_birth_weight'

    def __init__(self):
        super().__init__(f'risk_factor.{LowBirthWeight.RISK_NAME}')

    @staticmethod
    def parse_description(description: str) -> pd.Interval:
        # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'
        endpoints = pd.Interval(*[float(val) for val in description.split(', [')[1].split(')')[0].split(', ')])
        return endpoints


class ShortGestation(LBWSGSubRisk):

    RISK_NAME = 'short_gestation'

    def __init__(self):
        super().__init__(f'risk_factor.{ShortGestation.RISK_NAME}')

    @staticmethod
    def parse_description(description: str) -> pd.Interval:
        # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'
        endpoints = pd.Interval(*[float(val) for val in description.split('- [')[1].split(')')[0].split(', ')])
        return endpoints


class LBWSGRiskEffect(RiskEffect):
    # calculate individual's RRs on simulant initialization and store in a column which is source of RR

    def __init__(self, target: str):
        super().__init__('risk_factor.low_birth_weight_and_short_gestation', target)
        self.lbwsg_exposure_pipeline_name = f'{data_keys.LBWSG.name}.exposure'
        self.low_birth_weight_pipeline_name = 'low_birth_weight.exposure'
        self.short_gestation_pipeline_name = 'short_gestation.exposure'
        self.early_neonatal_relative_risk_column_name = (f'effect_of_{self.risk.name}_on_early_neonatal'
                                                         f'_{self.target.name}_relative_risk')
        self.late_neonatal_relative_risk_column_name = (f'effect_of_{self.risk.name}_on_late_neonatal'
                                                        f'_{self.target.name}_relative_risk')
        self.relative_risk_pipeline_name = f'effect_of_{self.risk.name}_on_{self.target.name}.relative_risk'

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.pipelines = self._get_pipelines(builder)
        self.population_view = self._get_population_view(builder)
        self.early_neonatal_interval, self.late_neonatal_interval = self._get_age_intervals(builder)
        self.interpolator = self._get_interpolator(builder)

        self._register_simulant_initializer(builder)

    def _get_relative_risk_source(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.relative_risk_pipeline_name,
            source=self._get_relative_risk,
            requires_columns=[self.early_neonatal_relative_risk_column_name,
                              self.late_neonatal_relative_risk_column_name]
        )

    def _get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        paf_data = builder.data.load(f'{self.risk}.population_attributable_fraction')
        return builder.lookup.build_table(paf_data, key_columns=['sex'], parameter_columns=['age', 'year'])

    def _get_target_modifier(self, builder: Builder) -> Callable[[pd.Index, pd.Series], pd.Series]:

        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            return target * self.relative_risk(index)

        return adjust_target

    def _get_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {
            self.lbwsg_exposure_pipeline_name: builder.value.get_value(self.lbwsg_exposure_pipeline_name),
            self.low_birth_weight_pipeline_name: builder.value.get_value(self.low_birth_weight_pipeline_name),
            self.short_gestation_pipeline_name: builder.value.get_value(self.short_gestation_pipeline_name)
        }

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([
            'age',
            'sex',
            self.early_neonatal_relative_risk_column_name,
            self.late_neonatal_relative_risk_column_name
        ])

    @staticmethod
    def _get_age_intervals(builder: Builder) -> Tuple[pd.Interval, pd.Interval]:
        age_bins = builder.data.load(data_keys.POPULATION.AGE_BINS).set_index('age_group_name')
        return tuple(
            pd.Interval(*(age_bins.loc[age_group_id, ['age_start', 'age_end']]))
            for age_group_id in ['Early Neonatal', 'Late Neonatal']
        )

    # noinspection PyMethodMayBeStatic
    def _get_interpolator(self, builder: Builder) -> pd.Series:
        # get relative risk data for target
        interpolators = builder.data.load(data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR)
        interpolators = (
            # isolate RRs for target and drop non-neonatal age groups since they have RR == 1.0
            interpolators[interpolators['age_end'] < 0.5]
            .drop(columns=['age_end', 'year_start', 'year_end'])
            .set_index(['sex', 'value'])
            .apply(lambda row: (metadata.AGE_GROUP.EARLY_NEONATAL_ID if row['age_start'] == 0.0
                                else metadata.AGE_GROUP.LATE_NEONATAL_ID), axis=1)
            .rename('age_group_id')
            .reset_index()
            .set_index(['sex', 'age_group_id'])
        )['value']

        interpolators = interpolators.apply(lambda x: pickle.loads(bytes.fromhex(x)))
        return interpolators

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.early_neonatal_relative_risk_column_name,
                             self.late_neonatal_relative_risk_column_name],
            requires_columns=['sex'],
            requires_values=[self.lbwsg_exposure_pipeline_name,
                             self.low_birth_weight_pipeline_name,
                             self.short_gestation_pipeline_name]
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        gestational_age = self.pipelines[self.short_gestation_pipeline_name](pop_data.index)
        birth_weight = self.pipelines[self.low_birth_weight_pipeline_name](pop_data.index)

        is_male = (
                self.population_view.subview(['sex']).get(pop_data.index).squeeze(axis=1) == 'Male'
        )
        is_tmrel = (
            (data_values.LBWSG.TMREL_GESTATIONAL_AGE_INTERVAL.left <= gestational_age)
            & (data_values.LBWSG.TMREL_BIRTH_WEIGHT_INTERVAL.left <= birth_weight)
        )

        def get_relative_risk_for_age_group(column_name: str, age_group_id: int) -> pd.Series:
            log_relative_risk = pd.Series(0.0, index=pop_data.index, name=column_name)
            log_relative_risk[is_male & ~is_tmrel] = (
                self.interpolator['Male', age_group_id](gestational_age[is_male & ~is_tmrel],
                                                        birth_weight[is_male & ~is_tmrel], grid=False)
            )
            log_relative_risk[~is_male & ~is_tmrel] = (
                self.interpolator['Female', age_group_id](gestational_age[~is_male & ~is_tmrel],
                                                          birth_weight[~is_male & ~is_tmrel], grid=False)
            )
            return np.exp(log_relative_risk)

        early_neonatal_rr = get_relative_risk_for_age_group(self.early_neonatal_relative_risk_column_name,
                                                            metadata.AGE_GROUP.EARLY_NEONATAL_ID)
        late_neonatal_rr = get_relative_risk_for_age_group(self.late_neonatal_relative_risk_column_name,
                                                           metadata.AGE_GROUP.LATE_NEONATAL_ID)
        self.population_view.update(pd.concat([early_neonatal_rr, late_neonatal_rr], axis=1))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_relative_risk(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        early_neonatal_mask = ((self.early_neonatal_interval.left <= pop.age)
                               & (pop.age < self.early_neonatal_interval.right))
        late_neonatal_mask = ((self.late_neonatal_interval.left <= pop.age)
                              & (pop.age < self.late_neonatal_interval.right))

        relative_risk = pd.Series(1.0, index=index)
        relative_risk[early_neonatal_mask] = pop.loc[early_neonatal_mask, self.early_neonatal_relative_risk_column_name]
        relative_risk[late_neonatal_mask] = pop.loc[late_neonatal_mask, self.late_neonatal_relative_risk_column_name]
        return relative_risk
