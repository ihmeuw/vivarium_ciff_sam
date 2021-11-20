from abc import abstractmethod, ABC
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.time import Time
from vivarium.framework.values import Pipeline

from vivarium_ciff_sam.constants import data_keys, data_values, scenarios


class LinearScaleUpIntervention(ABC):

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        self.is_intervention_scenario = self.get_is_intervention_scenario(builder)
        self.clock = self.get_clock(builder)
        self.scale_up_start_date, self.scale_up_end_date = self.get_scale_up_date_endpoints()
        self.scale_up_start_value, self.scale_up_end_value = self.get_scale_up_value_endpoints(builder)

        required_columns = self.get_required_columns()
        self.pipelines = self.get_required_pipelines(builder)

        self.register_intervention_modifiers(builder)

        if required_columns:
            self.population_view = self.get_population_view(builder, required_columns)

    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        return builder.configuration.intervention.scenario != 'baseline'

    # noinspection PyMethodMayBeStatic
    def get_clock(self, builder: Builder) -> Callable[[], Time]:
        return builder.time.clock()

    # noinspection PyMethodMayBeStatic
    def get_scale_up_date_endpoints(self) -> Tuple[datetime, datetime]:
        return data_values.SCALE_UP_START_DT, data_values.SCALE_UP_END_DT

    @abstractmethod
    def get_scale_up_value_endpoints(self, builder: Builder) -> Tuple[LookupTable, LookupTable]:
        pass

    def get_required_columns(self) -> List[str]:
        return []

    def get_required_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {}

    def register_intervention_modifiers(self, builder: Builder):
        pass

    # noinspection PyMethodMayBeStatic
    def get_population_view(self, builder: Builder, required_columns: List[str]):
        return builder.population.get_view(required_columns)

    @abstractmethod
    def apply_scale_up(self, idx: pd.Index, target: pd.Series, scale_up_progress: float) -> pd.Series:
        pass

    def coverage_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        if not self.is_intervention_scenario or self.clock() < self.scale_up_start_date:
            scale_up_progress = 0.0
        elif self.scale_up_start_date <= self.clock() < self.scale_up_end_date:
            scale_up_progress = ((self.clock() - self.scale_up_start_date)
                                 / (self.scale_up_end_date - self.scale_up_start_date))
        else:
            scale_up_progress = 1.0

        target = self.apply_scale_up(idx, target, scale_up_progress) if scale_up_progress else target
        return target


class SQLNSIntervention(LinearScaleUpIntervention):

    def __init__(self):
        self.name = 'sqlns_intervention'
        self.sqlns_propensity_pipeline_name = data_keys.SQ_LNS.PROPENSITY_PIPELINE
        self.sqlns_coverage_pipeline_name = data_keys.SQ_LNS.COVERAGE_PIPELINE

    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        return scenarios.SCENARIOS[builder.configuration.intervention.scenario].has_sqlns

    def get_scale_up_value_endpoints(self, builder: Builder) -> Tuple[LookupTable, LookupTable]:
        return (builder.lookup.build_table(data_values.SQ_LNS.COVERAGE_BASELINE),
                builder.lookup.build_table(data_values.SQ_LNS.COVERAGE_RAMP_UP))

    def get_required_columns(self) -> List[str]:
        return ["age"]

    # noinspection PyMethodMayBeStatic
    def get_required_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {self.sqlns_propensity_pipeline_name: builder.value.get_value(self.sqlns_propensity_pipeline_name)}

    def register_intervention_modifiers(self, builder: Builder):
        builder.value.register_value_modifier(
            self.sqlns_coverage_pipeline_name,
            modifier=self.coverage_effect,
            requires_columns=['age'],
            requires_values=[self.sqlns_propensity_pipeline_name]
        )

    def apply_scale_up(self, idx: pd.Index, target: pd.Series, scale_up_progress: float) -> pd.Series:
        age = self.population_view.get(idx)['age']
        propensity = self.pipelines[self.sqlns_propensity_pipeline_name](idx)
        start_value = self.scale_up_start_value(idx)
        end_value = self.scale_up_end_value(idx)

        effect = ((data_values.SQ_LNS.COVERAGE_START_AGE <= age)
                  & (propensity < start_value + (end_value - start_value) * scale_up_progress))
        return target | effect


class WastingTreatmentIntervention(LinearScaleUpIntervention):

    def __init__(self, wasting_treatment: str):
        self.name = f'{wasting_treatment}_intervention'
        self.treatment = {
            data_keys.SAM_TREATMENT.name: data_keys.SAM_TREATMENT,
            data_keys.MAM_TREATMENT.name: data_keys.MAM_TREATMENT
        }[wasting_treatment]

    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        return scenarios.SCENARIOS[builder.configuration.intervention.scenario].has_alternative_wasting_treatment

    def get_scale_up_value_endpoints(self, builder: Builder) -> Tuple[LookupTable, LookupTable]:
        baseline_coverage = builder.data.load(self.treatment.EXPOSURE)
        baseline_coverage = (
            baseline_coverage[baseline_coverage['parameter'] == self.treatment.BASELINE_COVERAGE]
            .drop(columns='parameter')
        )
        return (builder.lookup.build_table(baseline_coverage, key_columns=['sex'], parameter_columns=['age', 'year']),
                builder.lookup.build_table(data_values.WASTING.ALTERNATIVE_TX_COVERAGE))

    def register_intervention_modifiers(self, builder: Builder):
        # NOTE: this operation is NOT commutative. This pipeline must not be modified in any other component.
        builder.value.register_value_modifier(
            f'risk_factor.{self.treatment.name}.exposure_parameters',
            modifier=self.coverage_effect,
        )

    def apply_scale_up(self, idx: pd.Index, target: pd.Series, scale_up_progress: float) -> pd.Series:
        start_value = self.scale_up_start_value(idx)
        end_value = self.scale_up_end_value(idx)

        coverage_value = scale_up_progress * (end_value - start_value) + start_value

        target[self.treatment.UNCOVERED] = 1 - coverage_value
        target[self.treatment.BASELINE_COVERAGE] = (1 - scale_up_progress) * coverage_value
        target[self.treatment.ALTERNATIVE_COVERAGE] = scale_up_progress * coverage_value
        return target
