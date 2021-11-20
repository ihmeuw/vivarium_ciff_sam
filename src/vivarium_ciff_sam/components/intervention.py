from abc import abstractmethod, ABC
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.time import Time, get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.utilities import EntityString

from vivarium_ciff_sam.constants import data_keys, data_values, scenarios


class LinearScaleUpIntervention(ABC):

    def __init__(self, treatment: str):
        self.treatment = EntityString(treatment)
        self.name = f'{self.treatment.name}_intervention'
        self.configuration_defaults = self.get_configuration_defaults()

    def get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "scale_up_interventions": {
                f"{self.treatment.name}": {
                    "start": 'sim_start',
                    "end": 'sim_end',
                    "start_value": 'data',
                    "end_value": 'data',
                }
            }
        }

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        self.is_intervention_scenario = self.get_is_intervention_scenario(builder)
        self.clock = self.get_clock(builder)
        self.scale_up_start_date, self.scale_up_end_date = self.get_scale_up_date_endpoints(builder)
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
    def get_scale_up_date_endpoints(self, builder: Builder) -> Tuple[datetime, datetime]:
        scale_up_config = builder.configuration.scale_up_interventions[f'{self.treatment.name}']

        def get_endpoint(endpoint_type: str) -> datetime:
            if scale_up_config.start == f'sim_{endpoint_type}':
                endpoint = get_time_stamp(builder.configuration.time[endpoint_type])
            else:
                endpoint = get_time_stamp(scale_up_config[endpoint_type])
            return endpoint

        return get_endpoint('start'), get_endpoint('end')

    def get_scale_up_value_endpoints(self, builder: Builder) -> Tuple[LookupTable, LookupTable]:
        scale_up_config = builder.configuration.scale_up_interventions[f'{self.treatment.name}']

        def get_endpoint_value(endpoint_type: str) -> LookupTable:
            if scale_up_config[endpoint_type] == 'data':
                endpoint = self.get_endpoint_value_from_data(builder, endpoint_type)
            else:
                endpoint = builder.lookup.build_table(scale_up_config[endpoint_type])
            return endpoint

        return get_endpoint_value('start_value'), get_endpoint_value('end_value')

    def get_endpoint_value_from_data(self, builder: Builder, endpoint_type: str) -> LookupTable:
        if endpoint_type == 'start_value':
            endpoint_data = builder.data.load(f'{self.treatment}.exposure')
        elif endpoint_type == 'end_value':
            endpoint_data = builder.data.load(f'alternate_{self.treatment}.exposure')
        else:
            raise ValueError(f'Invalid endpoint type {endpoint_type}. Allowed types are "start_value" and "end_value".')
        return builder.lookup.build_table(endpoint_data)

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
        super().__init__('treatment.sqlns')
        self.sqlns_propensity_pipeline_name = data_keys.SQ_LNS.PROPENSITY_PIPELINE
        self.sqlns_coverage_pipeline_name = data_keys.SQ_LNS.COVERAGE_PIPELINE

    def get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "scale_up_interventions": {
                f"{self.treatment.name}": {
                    "start": {
                        "year": data_values.SCALE_UP_START_DT.year,
                        "month": data_values.SCALE_UP_START_DT.month,
                        "day": data_values.SCALE_UP_START_DT.day,
                    },
                    "end": {
                        "year": data_values.SCALE_UP_END_DT.year,
                        "month": data_values.SCALE_UP_END_DT.month,
                        "day": data_values.SCALE_UP_END_DT.day,
                    },
                    "start_value": data_values.SQ_LNS.COVERAGE_BASELINE,
                    "end_value": data_values.SQ_LNS.COVERAGE_RAMP_UP,
                }
            }
        }

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

    def __init__(self, treatment: str):
        super().__init__(treatment)

        self.treatment_keys = {
            data_keys.SAM_TREATMENT.name: data_keys.SAM_TREATMENT,
            data_keys.MAM_TREATMENT.name: data_keys.MAM_TREATMENT
        }[self.treatment.name]

    def get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "scale_up_interventions": {
                f"{self.treatment.name}": {
                    "start": {
                        "year": data_values.SCALE_UP_START_DT.year,
                        "month": data_values.SCALE_UP_START_DT.month,
                        "day": data_values.SCALE_UP_START_DT.day,
                    },
                    "end": {
                        "year": data_values.SCALE_UP_END_DT.year,
                        "month": data_values.SCALE_UP_END_DT.month,
                        "day": data_values.SCALE_UP_END_DT.day,
                    },
                    "start_value": 'data',
                    "end_value": data_values.WASTING.ALTERNATIVE_TX_COVERAGE,
                }
            }
        }

    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        return scenarios.SCENARIOS[builder.configuration.intervention.scenario].has_alternative_wasting_treatment

    def get_endpoint_value_from_data(self, builder: Builder, endpoint_type: str) -> LookupTable:
        if endpoint_type != 'start_value':
            raise ValueError(f'Invalid endpoint type {endpoint_type}. "start_value" is the only allowed type.')

        baseline_coverage = builder.data.load(self.treatment_keys.EXPOSURE)
        baseline_coverage = (
            baseline_coverage[baseline_coverage['parameter'] == self.treatment_keys.BASELINE_COVERAGE]
            .drop(columns='parameter')
        )
        return builder.lookup.build_table(baseline_coverage, key_columns=['sex'], parameter_columns=['age', 'year'])

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

        target[self.treatment_keys.UNCOVERED] = 1 - coverage_value
        target[self.treatment_keys.BASELINE_COVERAGE] = (1 - scale_up_progress) * coverage_value
        target[self.treatment_keys.ALTERNATIVE_COVERAGE] = scale_up_progress * coverage_value
        return target
