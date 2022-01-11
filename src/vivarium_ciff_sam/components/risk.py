from typing import Callable

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    pivot_categorical,
    rebin_relative_risk_data
)
from vivarium_public_health.risks.distributions import SimulationDistribution


class RiskWithTracked(Risk):

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name, 'tracked'])


class MaternalSupplementation(RiskWithTracked):

    def __init__(self):
        super().__init__('risk_factor.maternal_supplementation')
        self._sub_components = []

    ##########################
    # Initialization methods #
    ##########################

    def _get_exposure_distribution(self) -> SimulationDistribution:
        return None

    #################
    # Setup methods #
    #################

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return None


class MaternalSupplementationType(Risk):

    def __init__(self, risk: str):
        super().__init__(risk)
        self.propensity_column_name = 'maternal_supplementation_propensity'
        self.exposure_column_name = f'{self.risk.name}_exposure'

    #################
    # Setup methods #
    #################

    def _get_randomness_stream(self, builder) -> RandomnessStream:
        return None

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [self.propensity_column_name, 'tracked', self.exposure_column_name]
        )

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.exposure_column_name],
            requires_values=[self.exposure_pipeline_name]
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        exposure = pd.Series(self.exposure(pop_data.index), name=self.exposure_column_name)
        self.population_view.update(exposure)


class AdditiveRiskEffect(RiskEffect):

    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.target_risk_specific_shift_pipeline_name = f'{self.target.name}.risk_specific_shift'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.excess_shift_source = self._get_excess_shift_source(builder)
        self.risk_specific_shift_source = self._get_risk_specific_shift_source(builder)
        self._register_risk_specific_shift_modifier(builder)

    # NOTE: this RR will never be used. Overriding superclass to avoid error
    def _get_relative_risk_source(self, builder: Builder) -> LookupTable:
        return builder.lookup.build_table(1)

    # NOTE: this PAF will never be used. Overriding superclass to avoid error
    def _get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        return builder.lookup.build_table(0)

    def _get_target_modifier(self, builder: Builder) -> Callable[[pd.Index, pd.Series], pd.Series]:
        risk_exposure = builder.value.get_value(f'{self.risk.name}.exposure')

        def exposure_effect(target, rr: pd.DataFrame) -> pd.Series:
            index_columns = ['index', self.risk.name]

            exposure = risk_exposure(rr.index).reset_index()
            exposure.columns = index_columns
            exposure = exposure.set_index(index_columns)

            relative_risk = rr.stack().reset_index()
            relative_risk.columns = index_columns + ['value']
            relative_risk = relative_risk.set_index(index_columns)

            effect = relative_risk.loc[exposure.index, 'value'].droplevel(self.risk.name)
            affected_rates = target + effect
            return affected_rates

        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            return exposure_effect(target, self.excess_shift_source(index))

        return adjust_target

    def _get_excess_shift_source(self, builder: Builder) -> LookupTable:
        excess_shift_data = builder.data.load(
            f'{self.risk}.excess_shift',
            affected_entity=self.target.name,
            affected_measure=self.target.measure
        )
        excess_shift_data = rebin_relative_risk_data(builder, self.risk, excess_shift_data)
        excess_shift_data = pivot_categorical(excess_shift_data)
        return builder.lookup.build_table(
            excess_shift_data,
            key_columns=['sex'],
            parameter_columns=['age', 'year']
        )

    def _get_risk_specific_shift_source(self, builder: Builder) -> LookupTable:
        risk_specific_shift_data = builder.data.load(
            f'{self.risk}.risk_specific_shift',
            affected_entity=self.target.name,
            affected_measure=self.target.measure
        )
        return builder.lookup.build_table(
            risk_specific_shift_data,
            key_columns=['sex'],
            parameter_columns=['age', 'year']
        )

    def _register_paf_modifier(self, builder: Builder) -> None:
        pass

    def _register_risk_specific_shift_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.target_risk_specific_shift_pipeline_name,
            modifier=self.risk_specific_shift_modifier,
            requires_columns=['age', 'sex']
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def risk_specific_shift_modifier(self, index: pd.Index, target: pd.Series) -> pd.Series:
        return target + self.risk_specific_shift_source(index)
