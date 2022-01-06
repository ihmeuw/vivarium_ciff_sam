from typing import Callable

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium_public_health.risks import RiskEffect

from vivarium_ciff_sam.constants import data_keys


class AdditiveRiskEffect(RiskEffect):

    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.target_risk_specific_shift_pipeline_name = f'{self.target.name}.risk_specific_shift'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.risk_specific_shift_source = self._get_risk_specific_shift_source(builder)
        self._register_risk_specific_shift_modifier(builder)

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
            return exposure_effect(target, self.relative_risk(index))

        return adjust_target

    def _get_risk_specific_shift_source(self, builder: Builder) -> LookupTable:
        risk_specific_shift_data = builder.data.load(
            data_keys.MATERNAL_MALNUTRITION.RISK_SPECIFIC_SHIFT,
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
