import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import get_exposure_post_processor
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.utilities import EntityString

from vivarium_ciff_sam.constants import data_keys, data_values, metadata, models


class RiskState(DiseaseState):

    def load_excess_mortality_rate_data(self, builder):
        # TODO update this when we add PEM
        return 0


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
    tmrel = SusceptibleState(models.WASTING_MODEL_NAME)
    mild = RiskState(
        models.MILD_WASTING_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mild_wasting_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
        }
    )
    moderate = RiskState(
        models.MODERATE_WASTING_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mam_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
        }
    )
    severe = RiskState(
        models.SEVERE_WASTING_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_sam_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
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
            'transition_rate': load_mild_wasting_remission_rate,
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

    return RiskModel(
        models.WASTING_MODEL_NAME,
        get_data_functions={'cause_specific_mortality_rate': lambda *_: 0},
        states=[tmrel, mild, moderate, severe]
    )


def load_child_wasting_exposures(builder: Builder) -> pd.DataFrame:
    exposures = (
        builder.data.load(data_keys.WASTING.EXPOSURE)
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
        .pivot(columns='parameter')
    )

    exposures.columns = exposures.columns.droplevel(0)

    return exposures


# noinspection PyUnusedLocal
def load_mild_wasting_exposure(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[data_keys.WASTING.MILD].reset_index()


# noinspection PyUnusedLocal, DuplicatedCode
def load_mild_wasting_incidence_rate(cause: str, builder: Builder) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    return calculate_mild_wasting_incidence_rate(exposures, builder.configuration.wasting_equations.include_mortality)


# noinspection DuplicatedCode
def calculate_mild_wasting_incidence_rate(exposures: pd.DataFrame, include_mortality: bool = True) -> pd.DataFrame:
    if include_mortality:
        # TODO
        i3 = 0
    else:
        # i3: (dur_cat1*dur_cat2*p3 - dur_cat1*dur_cat3*p2 + dur_cat2*dur_cat3*p1)/(dur_cat1*dur_cat2*dur_cat3*p4)
        i3 = ((data_values.SAM_DURATION * data_values.MAM_DURATION * exposures[data_keys.WASTING.MILD]
               - data_values.SAM_DURATION * data_values.MILD_WASTING_DURATION * exposures[data_keys.WASTING.MAM]
               + data_values.MAM_DURATION * data_values.MILD_WASTING_DURATION * exposures[data_keys.WASTING.SAM])
              / (data_values.SAM_DURATION * data_values.MAM_DURATION * data_values.MILD_WASTING_DURATION
                 * exposures[data_keys.WASTING.TMREL])).reset_index()
    return i3


# noinspection PyUnusedLocal
def load_mild_wasting_remission_rate(builder: Builder, *args) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    return calculate_mild_wasting_remission_rate(exposures, builder.configuration.wasting_equations.include_mortality)


# noinspection DuplicatedCode
def calculate_mild_wasting_remission_rate(exposures: pd.DataFrame, include_mortality: bool = True) -> pd.DataFrame:
    if include_mortality:
        # TODO
        r4 = 0
    else:
        # r4: (dur_cat1*dur_cat2*p3 - dur_cat1*dur_cat3*p2 + dur_cat2*dur_cat3*p1)/(dur_cat1*dur_cat2*dur_cat3*p3)
        r4 = ((data_values.SAM_DURATION * data_values.MAM_DURATION * exposures[data_keys.WASTING.MILD]
               - data_values.SAM_DURATION * data_values.MILD_WASTING_DURATION * exposures[data_keys.WASTING.MAM]
               + data_values.MAM_DURATION * data_values.MILD_WASTING_DURATION * exposures[data_keys.WASTING.SAM])
              / (data_values.SAM_DURATION * data_values.MAM_DURATION * data_values.MILD_WASTING_DURATION
                 * exposures[data_keys.WASTING.MILD])).reset_index()
    return r4


# noinspection PyUnusedLocal
def load_mam_exposure(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[data_keys.WASTING.MAM].reset_index()


# noinspection PyUnusedLocal
def load_mam_incidence_rate(builder: Builder, *args) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    return calculate_mam_incidence_rate(exposures, builder.configuration.wasting_equations.include_mortality)


# noinspection DuplicatedCode
def calculate_mam_incidence_rate(exposures: pd.DataFrame, include_mortality: bool = True) -> pd.DataFrame:
    if include_mortality:
        # TODO
        i2 = 0
    else:
        # i2: (dur_cat1*p2 - dur_cat2*p1)/(dur_cat1*dur_cat2*p3)
        i2 = ((data_values.SAM_DURATION * exposures[data_keys.WASTING.MAM])
              - (data_values.MAM_DURATION * exposures[data_keys.WASTING.SAM])
              / (data_values.SAM_DURATION * data_values.MAM_DURATION * exposures[data_keys.WASTING.MILD])
              ).reset_index()
    return i2


# noinspection PyUnusedLocal
def load_mam_remission_rate(builder: Builder, *args) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    return calculate_mam_remission_rate(exposures, builder.configuration.wasting_equations.include_mortality)


# noinspection DuplicatedCode
def calculate_mam_remission_rate(exposures: pd.DataFrame, include_mortality: bool = True) -> pd.DataFrame:
    if include_mortality:
        # TODO
        r3 = 0
    else:
        # r3: (dur_cat1*p2 - dur_cat2*p1)/(dur_cat1*dur_cat2*p2)
        r3 = ((data_values.SAM_DURATION * exposures[data_keys.WASTING.MAM])
              - (data_values.MAM_DURATION * exposures[data_keys.WASTING.SAM])
              / (data_values.SAM_DURATION * data_values.MAM_DURATION * exposures[data_keys.WASTING.MAM])
              ).reset_index()
    return r3


# noinspection PyUnusedLocal
def load_sam_exposure(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[data_keys.WASTING.SAM].reset_index()


# noinspection PyUnusedLocal
def load_sam_incidence_rate(builder: Builder, *args) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    return calculate_sam_incidence_rate(exposures, builder.configuration.wasting_equations.include_mortality)


def calculate_sam_incidence_rate(exposures: pd.DataFrame, include_mortality: bool = True) -> pd.DataFrame:
    if include_mortality:
        # TODO
        i1 = 0
    else:
        # i1: p1/(dur_cat1*p2)
        i1 = (exposures[data_keys.WASTING.SAM]
              / (data_values.SAM_DURATION * exposures[data_keys.WASTING.MAM])).reset_index()
    return i1


# noinspection PyUnusedLocal
def load_sam_remission_rate(builder: Builder, *args) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    return calculate_sam_remission_rate(builder.configuration.wasting_equations.include_mortality)


def calculate_sam_remission_rate(include_mortality: bool = True) -> pd.DataFrame:
    if include_mortality:
        # TODO
        r2 = 0
    else:
        # r2: 1/dur_cat1
        r2 = 1 / data_values.SAM_DURATION
    return r2
