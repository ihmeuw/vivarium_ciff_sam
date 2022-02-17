from typing import Callable, Dict, List

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
    pivot_categorical,
    rebin_relative_risk_data
)
from vivarium_public_health.risks.distributions import SimulationDistribution

from vivarium_ciff_sam.constants import data_keys


class RiskWithTracked(Risk):

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name, 'tracked'])


class MaternalSupplementation(Risk):

    def __init__(self):
        super().__init__('risk_factor.maternal_supplementation')
        self._sub_components = []

        self.ifa_exposure_column_name = f'{data_keys.IFA_SUPPLEMENTATION.name}_exposure'
        self.mmn_exposure_column_name = f'{data_keys.MMN_SUPPLEMENTATION.name}_exposure'
        self.bep_exposure_column_name = f'{data_keys.BEP_SUPPLEMENTATION.name}_exposure'

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

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self._get_current_exposure,
            requires_columns=[
                'tracked',
                self.ifa_exposure_column_name,
                self.mmn_exposure_column_name,
                self.bep_exposure_column_name,
            ],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [
                self.propensity_column_name,
                'tracked',
                self.ifa_exposure_column_name,
                self.mmn_exposure_column_name,
                self.bep_exposure_column_name,
            ]
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index: pd.Index) -> pd.Series:
        required_columns = [
            'tracked',
            self.ifa_exposure_column_name,
            self.mmn_exposure_column_name,
            self.bep_exposure_column_name
        ]
        pop = self.population_view.get(index)[required_columns]
        has_bep = pop[self.bep_exposure_column_name] == data_keys.BEP_SUPPLEMENTATION.CAT2
        has_mmn = pop[self.mmn_exposure_column_name] == data_keys.MMN_SUPPLEMENTATION.CAT2
        has_ifa = pop[self.ifa_exposure_column_name] == data_keys.IFA_SUPPLEMENTATION.CAT2

        exposure = pd.Series('uncovered', index=index)
        exposure[has_ifa] = 'ifa'
        exposure[has_mmn] = 'mmn'
        exposure[has_bep] = 'bep'
        return exposure


class BirthWeightIntervention(Risk):

    def __init__(self, risk: str):
        super().__init__(risk)
        self.exposure_column_name = f'{self.risk.name}_exposure'

    #################
    # Setup methods #
    #################

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self._get_current_exposure,
            requires_columns=[self.exposure_column_name],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [self.propensity_column_name, 'tracked', self.exposure_column_name]
        )

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.exposure_column_name, self.propensity_column_name],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        propensity = pd.Series(
            self.randomness.get_draw(pop_data.index), name=self.propensity_column_name
        )

        exposure = pd.Series(
            self.exposure_distribution.ppf(propensity),
            index=pop_data.index,
            name=self.exposure_column_name
        )
        self.population_view.update(pd.concat([propensity, exposure], axis=1))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index: pd.Index) -> pd.Series:
        exposure = (
            self.population_view
            .subview([self.exposure_column_name])
            .get(index)
            .squeeze(axis=1)
        )
        return exposure


class MaternalSupplementationType(BirthWeightIntervention):

    def __init__(self, risk: str):
        super().__init__(risk)
        self.propensity_column_name = 'maternal_supplementation_propensity'

    #################
    # Setup methods #
    #################

    def _get_randomness_stream(self, builder) -> RandomnessStream:
        return None

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.exposure_column_name],
            requires_values=[self.propensity_pipeline_name]
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        propensity = self.propensity(pop_data.index)
        exposure = pd.Series(
            self.exposure_distribution.ppf(propensity),
            index=pop_data.index,
            name=self.exposure_column_name
        )
        self.population_view.update(exposure)


class BEPSupplementation(BirthWeightIntervention):

    def __init__(self):
        super().__init__(f'risk_factor.{data_keys.BEP_SUPPLEMENTATION.name}')
        self.maternal_malnutrition_exposure_pipeline_name = (
            f'{data_keys.MATERNAL_MALNUTRITION.name}.exposure'
        )
        self.mmn_exposure_pipeline_name = f'{data_keys.MMN_SUPPLEMENTATION.name}.exposure'

    ##########################
    # Initialization methods #
    ##########################

    def _get_exposure_distribution(self) -> SimulationDistribution:
        return None

    ##############
    # Properties #
    ##############

    @property
    def sub_components(self) -> List:
        return []

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.pipelines = self._get_pipelines(builder)

    def _get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return None

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(['tracked', self.exposure_column_name])

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.exposure_column_name],
            requires_values=[
                self.maternal_malnutrition_exposure_pipeline_name,
                self.mmn_exposure_pipeline_name,
            ]
        )

    def _get_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {
            self.maternal_malnutrition_exposure_pipeline_name:
                builder.value.get_value(self.maternal_malnutrition_exposure_pipeline_name),
            self.mmn_exposure_pipeline_name:
                builder.value.get_value(self.mmn_exposure_pipeline_name)
        }

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        maternal_malnutrition_exposure = (
            self.pipelines[self.maternal_malnutrition_exposure_pipeline_name](pop_data.index)
        )
        mmn_exposure = self.pipelines[self.mmn_exposure_pipeline_name](pop_data.index)
        bep_exposure_mask = (
            (maternal_malnutrition_exposure == data_keys.MATERNAL_MALNUTRITION.CAT1)
            & (mmn_exposure == data_keys.MMN_SUPPLEMENTATION.CAT2)
        )

        exposure = pd.Series(
            data_keys.BEP_SUPPLEMENTATION.CAT1,
            index=pop_data.index,
            name=self.exposure_column_name
        )
        exposure[bep_exposure_mask] = data_keys.BEP_SUPPLEMENTATION.CAT2
        self.population_view.update(exposure)


class PreventativeZincSupplementation(Risk):

    def __init__(self, risk: str):
        super().__init__(risk)
        self.propensity_column_name = f'therapeutic_zinc_propensity'

    #################
    # Setup methods #
    #################

    def _register_simulant_initializer(self, builder: Builder) -> None:
        # builder.population.initializes_simulants(
        #     self.on_initialize_simulants
        # )
        pass

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
