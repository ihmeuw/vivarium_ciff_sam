from typing import Union

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import get_exposure_post_processor
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.utilities import EntityString

from vivarium_ciff_sam.constants import data_keys, data_values, metadata, models
from vivarium_ciff_sam.constants.data_keys import WASTING


class RiskState(DiseaseState):

    def load_excess_mortality_rate_data(self, builder):
        if 'excess_mortality_rate' in self._get_data_functions:
            return self._get_data_functions['excess_mortality_rate'](self.cause, builder)
        else:
            return builder.data.load(f'{self.cause_type}.{self.cause}.excess_mortality_rate')


class RiskModel(DiseaseModel):

    def __init__(self, risk, **kwargs):
        super().__init__(risk, **kwargs)
        self.configuration_defaults = {f'{self.state_column}': Risk.configuration_defaults['risk']}

    # This would be a preferable name, but the generic DiseaseObserver works with no modifications if we use the
    # standard naming from DiseaseModel. Extending to DiseaseObserver to RiskObserver would provide no functional gain
    # and involve copy-pasting a bunch of code
    # @property
    # def name(self):
    #     return f"risk_model.{self.state_column}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder):
        super().setup(builder)

        self.exposure = builder.value.register_value_producer(
            f'{self.state_column}.exposure',
            source=self.get_current_exposure,
            requires_columns=['age', 'sex', self.state_column],
            preferred_post_processor=get_exposure_post_processor(builder,
                                                                 EntityString(f'risk_factor.{self.state_column}'))
        )

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        wasting_state = self.population_view.subview([self.state_column]).get(index)[self.state_column]
        return wasting_state.apply(models.get_risk_category)


# noinspection PyPep8Naming
def ChildWasting():
    tmrel = SusceptibleState(models.WASTING.MODEL_NAME)
    mild = RiskState(
        models.WASTING.MILD_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mild_wasting_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
            'birth_prevalence': load_mild_wasting_birth_prevalence,
        }
    )
    moderate = RiskState(
        models.WASTING.MODERATE_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mam_exposure,
            'excess_mortality_rate': load_pem_excess_mortality_rate,
            'birth_prevalence': load_mam_birth_prevalence,
        }
    )
    severe = RiskState(
        models.WASTING.SEVERE_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_sam_exposure,
            'excess_mortality_rate': load_pem_excess_mortality_rate,
            'birth_prevalence': load_sam_birth_prevalence,
        }
    )

    # Add transitions for tmrel
    tmrel.allow_self_transitions()
    tmrel.add_transition(
        mild,
        source_data_type='rate',
        get_data_functions={
            'incidence_rate': load_mild_wasting_incidence_rate,
        }
    )

    # Add transitions for mild
    mild.allow_self_transitions()
    mild.add_transition(
        moderate,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_mam_incidence_rate,
        }
    )
    mild.add_transition(
        tmrel,
        source_data_type='rate',
        get_data_functions={
            'remission_rate': load_mild_wasting_remission_rate,
        }
    )

    # Add transitions for moderate
    moderate.allow_self_transitions()
    moderate.add_transition(
        severe,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_sam_incidence_rate,
        }
    )
    moderate.add_transition(
        mild,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_mam_remission_rate,
        }
    )

    # Add transitions for severe
    severe.allow_self_transitions()
    severe.add_transition(
        moderate,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_sam_remission_rate,
        }
    )
    severe.add_transition(
        mild,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_sam_treated_remission_rate,
        }
    )

    return RiskModel(
        models.WASTING.MODEL_NAME,
        get_data_functions={'cause_specific_mortality_rate': lambda *_: 0},
        states=[tmrel, mild, moderate, severe]
    )


# noinspection PyUnusedLocal
def load_pem_excess_mortality_rate(cause: str, builder: Builder) -> pd.DataFrame:
    return builder.data.load(data_keys.PEM.EMR)


# noinspection PyUnusedLocal
def load_mild_wasting_birth_prevalence(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT3)


# noinspection PyUnusedLocal
def load_mild_wasting_exposure(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT3].reset_index()


# noinspection PyUnusedLocal, DuplicatedCode
def load_mild_wasting_incidence_rate(cause: str, builder: Builder) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    adjustment = load_acmr_adjustment(builder)
    mortality_probs = load_daily_mortality_probabilities(builder)

    daily_probability = get_daily_mild_incidence_probability(exposures, adjustment, mortality_probs)
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


# noinspection DuplicatedCode
def get_daily_mild_incidence_probability(exposures: pd.DataFrame, adjustment: pd.Series,
                                         mortality_probs: pd.DataFrame) -> pd.Series:
    adj_exposures = adjust_exposure(exposures, adjustment)
    mild_remission_prob = get_mild_wasting_remission_probability()

    # i3: ap0*f4/ap4 + ap3*r4/ap4 - d4
    i3 = (
            adjustment * exposures[WASTING.CAT4] / adj_exposures[WASTING.CAT4]
            + adj_exposures[WASTING.CAT3] * mild_remission_prob / adj_exposures[WASTING.CAT4]
            - mortality_probs[WASTING.CAT4]
    )
    return i3


# noinspection PyUnusedLocal
def load_mild_wasting_remission_rate(builder: Builder, *args) -> float:
    daily_probability = get_mild_wasting_remission_probability()
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate


# noinspection DuplicatedCode
def get_mild_wasting_remission_probability() -> float:
    return 1 / data_values.MILD_WASTING_UX_RECOVERY_TIME


# noinspection PyUnusedLocal
def load_mam_birth_prevalence(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT2)


# noinspection PyUnusedLocal
def load_mam_exposure(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT2].reset_index()


# noinspection PyUnusedLocal
def load_mam_incidence_rate(builder: Builder, *args) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    adjustment = load_acmr_adjustment(builder)
    mortality_probs = load_daily_mortality_probabilities(builder)

    daily_probability = get_daily_mam_incidence_probability(exposures, adjustment, mortality_probs)
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


# noinspection DuplicatedCode
def get_daily_mam_incidence_probability(exposures: pd.DataFrame, adjustment: pd.Series,
                                        mortality_probs: pd.DataFrame) -> pd.Series:
    adj_exposures = adjust_exposure(exposures, adjustment)
    treated_sam_remission_prob = get_daily_sam_treated_remission_probability()
    mam_remission_prob = get_daily_mam_remission_probability()

    # i2: ap0*f3/ap3 + ap0*f4/ap3 + ap1*t1/ap3 + ap2*r3/ap3 - d3 - ap4*d4/ap3
    i2 = (
            adjustment * exposures[WASTING.CAT3] / adj_exposures[WASTING.CAT3]
            + adjustment * exposures[WASTING.CAT4] / adj_exposures[WASTING.CAT3]
            + adj_exposures[WASTING.CAT1] * treated_sam_remission_prob / adj_exposures[WASTING.CAT3]
            + adj_exposures[WASTING.CAT2] * mam_remission_prob / adj_exposures[WASTING.CAT3]
            - mortality_probs[WASTING.CAT3]
            - adj_exposures[WASTING.CAT4] * mortality_probs[WASTING.CAT4] / adj_exposures[WASTING.CAT3]
    )
    return i2


# noinspection PyUnusedLocal
def load_mam_remission_rate(builder: Builder, *args) -> float:
    daily_probability = get_daily_mam_remission_probability()
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate


def get_daily_mam_remission_probability() -> float:
    # r3: mam_tx_coverage * 1/time_to_mam_tx_recovery + (1-mam_tx_coverage)*(1/time_to_mam_ux_recovery)
    r3 = (
        data_values.MAM_TX_COVERAGE / data_values.MAM_TX_RECOVERY_TIME
        + (1 - data_values.MAM_TX_COVERAGE) / data_values.MAM_UX_RECOVERY_TIME
    )
    return r3


# noinspection PyUnusedLocal
def load_sam_birth_prevalence(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT1)


# noinspection PyUnusedLocal
def load_sam_exposure(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT1].reset_index()


# noinspection PyUnusedLocal
def load_sam_incidence_rate(builder: Builder, *args) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    adjustment = load_acmr_adjustment(builder)
    mortality_probs = load_daily_mortality_probabilities(builder)

    daily_probability = get_daily_sam_incidence_probability(exposures, adjustment, mortality_probs)
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


def get_daily_sam_incidence_probability(exposures: pd.DataFrame, adjustment: pd.Series,
                                        mortality_probs: pd.DataFrame) -> pd.Series:
    adj_exposures = adjust_exposure(exposures, adjustment)
    treated_sam_remission_prob = get_daily_sam_treated_remission_probability()
    untreated_sam_remission_prob = get_daily_sam_untreated_remission_probability()

    # i1: ap0*f2/ap2 + ap0*f3/ap2 + ap0*f4/ap2 + ap1*r2/ap2 + ap1*t1/ap2 - d2 - ap3*d3/ap2 - ap4*d4/ap2
    i1 = (
            adjustment * exposures[WASTING.CAT2] / adj_exposures[WASTING.CAT2]
            + adjustment * exposures[WASTING.CAT3] / adj_exposures[WASTING.CAT2]
            + adjustment * exposures[WASTING.CAT4] / adj_exposures[WASTING.CAT2]
            + adj_exposures[WASTING.CAT1] * untreated_sam_remission_prob / adj_exposures[WASTING.CAT2]
            + adj_exposures[WASTING.CAT1] * treated_sam_remission_prob / adj_exposures[WASTING.CAT2]
            - mortality_probs[WASTING.CAT2]
            - adj_exposures[WASTING.CAT3] * mortality_probs[WASTING.CAT3] / adj_exposures[WASTING.CAT2]
            - adj_exposures[WASTING.CAT4] * mortality_probs[WASTING.CAT4] / adj_exposures[WASTING.CAT2]
    )
    return i1


# noinspection PyUnusedLocal
def load_sam_remission_rate(builder: Builder, *args) -> float:
    daily_probability = get_daily_sam_untreated_remission_probability()
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate


def get_daily_sam_untreated_remission_probability() -> float:
    # r2: (1-sam_tx_coverage)*(1/time_to_sam_ux_recovery)
    r2 = (1 - data_values.SAM_TX_COVERAGE) / data_values.SAM_UX_RECOVERY_TIME
    return r2


# noinspection PyUnusedLocal
def load_sam_treated_remission_rate(builder: Builder, *args) -> float:
    daily_probability = get_daily_sam_treated_remission_probability()
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate


def get_daily_sam_treated_remission_probability() -> float:
    # t1: sam_tx_coverage * (1/time_to_sam_tx_recovery)
    t1 = data_values.SAM_TX_COVERAGE / data_values.SAM_TX_RECOVERY_TIME
    return t1


# Sub-loader functions

def load_child_wasting_exposures(builder: Builder) -> pd.DataFrame:
    exposures = (
        builder.data.load(WASTING.EXPOSURE)
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
        .pivot(columns='parameter')
    )

    exposures.columns = exposures.columns.droplevel(0)
    return exposures


def load_child_wasting_birth_prevalence(builder: Builder, wasting_category: str) -> pd.DataFrame:
    exposure = load_child_wasting_exposures(builder)[wasting_category]
    birth_prevalence = (
        exposure[exposure.index.get_level_values('age_start') == 0.0]
        .droplevel(['age_start', 'age_end'])
        .reset_index()
    )
    return birth_prevalence


def load_acmr_adjustment(builder: Builder) -> pd.Series:
    acmr = (
        builder.data.load(data_keys.POPULATION.ACMR)
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
    )['value']

    adjustment = _convert_annual_rate_to_daily_probability(acmr)
    return adjustment


def load_daily_mortality_probabilities(builder: Builder) -> pd.DataFrame:
    """"
    Returns a DataFrame with daily mortality probabilities for each wasting state

    DataFrame has the standard artifact index, and columns for each wasting state
    """

    # ---------- Load mortality rate input data ---------- #
    causes = [
        data_keys.DIARRHEA,
        data_keys.MEASLES,
        data_keys.LRI,
        data_keys.PEM,
    ]

    # acmr
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end' ]
    acmr = builder.data.load(data_keys.POPULATION.ACMR).set_index(metadata.ARTIFACT_INDEX_COLUMNS)['value']

    # emr_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    emr_c = pd.concat(
        [builder.data.load(c.EMR).set_index(metadata.ARTIFACT_INDEX_COLUMNS).rename(columns={'value': c.name})
         for c in causes], axis=1
    )
    emr_c.columns.name = 'affected_entity'
    emr_c = emr_c.stack()

    # csmr_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    csmr_c = pd.concat(
        [builder.data.load(c.CSMR).set_index(metadata.ARTIFACT_INDEX_COLUMNS).rename(columns={'value': c.name})
         for c in causes], axis=1
    )
    csmr_c.columns.name = 'affected_entity'
    csmr_c = csmr_c.stack()

    # incidence_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    incidence_c = pd.concat(
        [builder.data.load(c.INCIDENCE_RATE).set_index(metadata.ARTIFACT_INDEX_COLUMNS)
         .rename(columns={'value': c.name}) for c in causes if c != data_keys.PEM], axis=1
    )
    incidence_c.columns.name = 'affected_entity'
    incidence_c = incidence_c.stack()

    # paf_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    paf_c = (builder.data.load(WASTING.PAF)
             .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ['affected_entity'])['value'])

    # rr_ci
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity', 'parameter ]
    rr_ci = (builder.data.load(WASTING.RELATIVE_RISK)
             .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ['affected_entity', 'parameter'])['value'])

    # duration_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity', 'parameter ]
    duration_c = pd.Series(
        [data_values.DIARRHEA_DURATION, data_values.MEASLES_DURATION, data_values.LRI_DURATION],
        index=pd.Index([data_keys.DIARRHEA.name, data_keys.MEASLES.name, data_keys.LRI.name], name='affected_entity')
    ).reindex(index=rr_ci.index, level='affected_entity')
    duration_c.loc[duration_c.index.get_level_values('age_start') == 0.0] = data_values.EARLY_NEONATAL_CAUSE_DURATION
    duration_c = duration_c / 365   # convert to duration in years

    # prevalence_pem_i
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity', 'parameter ]
    prevalence_pem_i = pd.DataFrame({'value': [1.0, 1.0, 0.0, 0.0], 'affected_entity': [data_keys.PEM.name]},
                                    index=pd.Index([f'cat{i}' for i in range(1, 5)], name='parameter'))
    prevalence_pem_i = (
        prevalence_pem_i.reindex(
            index=rr_ci.index.droplevel('affected_entity').drop_duplicates(),
            level='parameter'
        ).set_index('affected_entity', append=True)
        .reorder_levels(rr_ci.index.names)
    )['value']

    # ------------ Calculate mortality rates ------------ #

    # mr_i = acmr + sum_c(emr_c * prevalence_ci - csmr_c)
    # prevalence_ci = incidence_ci * duration_c
    # incidence_ci = incidence_c * (1 - paf_c) * rr_ci

    # Get wasting state incidence and prevalence for non-PEM causes
    incidence_ci = rr_ci * incidence_c * (1 - paf_c)
    prevalence_ci = incidence_ci * duration_c

    # add pem prevalence to prevalence_ci
    prevalence_ci = pd.concat([prevalence_ci, prevalence_pem_i])

    mr_i = acmr + (prevalence_ci * emr_c - csmr_c).groupby(metadata.ARTIFACT_INDEX_COLUMNS + ['parameter']).sum()
    mr_i = mr_i.unstack()

    # Convert annual mortality rates to daily mortality probabilities
    daily_mortality_probability = _convert_annual_rate_to_daily_probability(mr_i)
    return daily_mortality_probability


def adjust_exposure(exposures: pd.DataFrame, adjustment: pd.Series) -> pd.DataFrame:
    return exposures.div(1 + adjustment, axis='index')


def _convert_annual_rate_to_daily_probability(rate: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    return 1 - np.exp(-rate / 365)


def _convert_daily_probability_to_annual_rate(probability: Union[pd.Series, float]) -> Union[pd.Series, float]:
    return -np.log(1 - probability) * 365
