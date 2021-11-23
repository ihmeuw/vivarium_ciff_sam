from typing import Dict, List

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.values import Pipeline
from vivarium_public_health.treatment import LinearScaleUpIntervention

from vivarium_ciff_sam.constants import data_keys, data_values, scenarios


class SQLNSIntervention(LinearScaleUpIntervention):

    def __init__(self):
        super().__init__('treatment.sqlns')
        self.sqlns_propensity_pipeline_name = data_keys.SQ_LNS.PROPENSITY_PIPELINE
        self.sqlns_coverage_pipeline_name = data_keys.SQ_LNS.COVERAGE_PIPELINE

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Dict]:
        return {
            f"{self.treatment.name}_scale_up": {
                "start": {
                    "date": {
                        "year": data_values.SCALE_UP_START_DT.year,
                        "month": data_values.SCALE_UP_START_DT.month,
                        "day": data_values.SCALE_UP_START_DT.day,
                    },
                    "value": data_values.SQ_LNS.COVERAGE_BASELINE,
                },
                "end": {
                    "date": {
                        "year": data_values.SCALE_UP_END_DT.year,
                        "month": data_values.SCALE_UP_END_DT.month,
                        "day": data_values.SCALE_UP_END_DT.day,
                    },
                    "value": data_values.SQ_LNS.COVERAGE_RAMP_UP,
                }
            }
        }

    #################
    # Setup methods #
    #################

    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        return scenarios.SCENARIOS[builder.configuration.intervention.scenario].has_sqlns

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

    ##################
    # Helper methods #
    ##################

    def apply_scale_up(self, idx: pd.Index, target: pd.Series, scale_up_progress: float) -> pd.Series:
        # todo simply use idx rather than target.index once sqlns treatment sub-classes Risk
        age = self.population_view.get(target.index)['age']
        propensity = self.pipelines[self.sqlns_propensity_pipeline_name](target.index)
        start_value = self.scale_up_start_value(target.index)
        end_value = self.scale_up_end_value(target.index)

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

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Dict]:
        return {
            f"{self.treatment.name}_scale_up": {
                "start": {
                    "date": {
                        "year": data_values.SCALE_UP_START_DT.year,
                        "month": data_values.SCALE_UP_START_DT.month,
                        "day": data_values.SCALE_UP_START_DT.day,
                    },
                    "value": 'data',
                },
                "end": {
                    "date": {
                        "year": data_values.SCALE_UP_END_DT.year,
                        "month": data_values.SCALE_UP_END_DT.month,
                        "day": data_values.SCALE_UP_END_DT.day,
                    },
                    "value": data_values.WASTING.ALTERNATIVE_TX_COVERAGE,
                }
            }
        }

    #################
    # Setup methods #
    #################

    def get_is_intervention_scenario(self, builder: Builder) -> bool:
        return scenarios.SCENARIOS[builder.configuration.intervention.scenario].has_alternative_wasting_treatment

    ##################
    # Helper methods #
    ##################

    def get_endpoint_value_from_data(self, builder: Builder, endpoint_type: str) -> LookupTable:
        if endpoint_type != 'start':
            raise ValueError(f'Invalid endpoint type {endpoint_type}. "start" is the only allowed type.')

        baseline_coverage = builder.data.load(self.treatment_keys.EXPOSURE)
        baseline_coverage = (
            baseline_coverage[baseline_coverage['parameter'] == self.treatment_keys.BASELINE_COVERAGE]
            .drop(columns='parameter')
        )
        return builder.lookup.build_table(baseline_coverage, key_columns=['sex'], parameter_columns=['age', 'year'])

    def apply_scale_up(self, idx: pd.Index, target: pd.Series, scale_up_progress: float) -> pd.Series:
        # NOTE: this operation is NOT commutative. This pipeline must not be modified in any other component.
        start_value = self.scale_up_start_value(idx)
        end_value = self.scale_up_end_value(idx)

        coverage_value = scale_up_progress * (end_value - start_value) + start_value

        target[self.treatment_keys.UNCOVERED] = 1 - coverage_value
        target[self.treatment_keys.BASELINE_COVERAGE] = (1 - scale_up_progress) * coverage_value
        target[self.treatment_keys.ALTERNATIVE_COVERAGE] = scale_up_progress * coverage_value
        return target
