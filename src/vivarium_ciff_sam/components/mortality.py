"""
========================
The Core Mortality Model
========================
Summary
=======
The mortality component models all cause mortality and allows for disease
models to contribute cause specific mortality. At each timestep the
currently "alive" population is subjected to a mortality event that uses
the mortality hazard data to reap simulants. A weighted probable cause of
death is used to pick a cause of death. The years of life lost are calculated
by subtracting the simulant's age from the population TMRLE and the population
is updated.
Pipelines Exposed
=================
 - cause_specific_mortality_rate
 - mortality_rate
 - all_causes.mortality_hazard
All cause mortality is read from the artifact (GBD). At setup cause specific
mortality is initialized to an empty table. As disease models are incorporated
they register as affecting cause specific mortality and their contributions
are reflected in the cause_specific_mortality_rate pipeline. This is population
level data.
The mortality component's mortality_rate pipeline reflects the
cause deleted mortality rate (ACMR - CSMR).
Finally, the mortality component exposes a mortality hazard pipeline and a
mortality hazard PAF pipeline (used internally). The cause specific rates are
summed and added to the cause deleted mortality rate. These values are multiplied
by 1 - PAF. The end product comprises the values in the mortality hazard pipeline.
"""
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.values import Pipeline, union_post_processor, list_combiner
from vivarium_public_health.population import Mortality as _Mortality

from vivarium_ciff_sam.constants import data_keys


class Mortality(_Mortality):

    def __init__(self):
        super().__init__()
        self.unmodeled_csmr_pipeline_name = 'affected_unmodeled.cause_specific_mortality_rate'
        self.unmodeled_csmr_paf_pipeline_name = f'{self.unmodeled_csmr_pipeline_name}.paf'
        self.all_cause_mortality_hazard_pipeline_name = 'all_causes.mortality_hazard'
        self.all_cause_mortality_hazard_paf_pipeline_name = f'{self.all_cause_mortality_hazard_pipeline_name}.paf'

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        super().setup(builder)
        self._raw_unmodeled_csmr = self._get_raw_unmodeled_csmr(builder)
        self.unmodeled_csmr = self._get_unmodeled_csmr(builder)
        self.unmodeled_csmr_paf = self._get_unmodeled_csmr_paf(builder)
        self.mortality_hazard = self._get_mortality_hazard(builder)
        self._mortality_hazard_paf = self._get_mortality_hazard_paf(builder)

    # noinspection PyMethodMayBeStatic
    def _get_raw_unmodeled_csmr(self, builder: Builder) -> LookupTable:
        raw_csmr = pd.DataFrame()
        for idx, cause in enumerate(data_keys.AFFECTED_UNMODELED_CAUSES):
            if 0 == idx:
                raw_csmr = builder.data.load(cause)
            else:
                raw_csmr.loc[:, 'value'] += builder.data.load(cause).value

        return builder.lookup.build_table(raw_csmr, key_columns=['sex'], parameter_columns=['age', 'year'])

    def _get_unmodeled_csmr(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.unmodeled_csmr_pipeline_name,
            source=self._get_unmodeled_csmr_source,
            requires_columns=['age', 'sex']
        )

    def _get_unmodeled_csmr_paf(self, builder: Builder) -> Pipeline:
        unmodeled_csmr_paf = builder.lookup.build_table(0)
        return builder.value.register_value_producer(
            self.unmodeled_csmr_paf_pipeline_name,
            source=lambda index: [unmodeled_csmr_paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor
        )

    def _get_mortality_hazard(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.all_cause_mortality_hazard_pipeline_name,
            source=self._get_mortality_hazard_source
        )

    def _get_mortality_hazard_paf(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.all_cause_mortality_hazard_paf_pipeline_name,
            source=lambda index: [pd.Series(0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index, query="alive =='alive'")
        mortality_hazard = self.mortality_hazard(pop.index)
        deaths = self.random.filter_for_rate(pop.index, mortality_hazard, additional_key='death')
        if not deaths.empty:
            cause_of_death_weights = self.mortality_rate(deaths).divide(mortality_hazard.loc[deaths], axis=0)
            cause_of_death = self.random.choice(deaths, cause_of_death_weights.columns, cause_of_death_weights,
                                                additional_key='cause_of_death')
            pop.loc[deaths, 'alive'] = 'dead'
            pop.loc[deaths, 'exit_time'] = event.time
            pop.loc[deaths, 'years_of_life_lost'] = self.life_expectancy(deaths)
            pop.loc[deaths, 'cause_of_death'] = cause_of_death
            self.population_view.update(pop)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _calculate_mortality_rate(self, index: pd.Index) -> pd.DataFrame:
        acmr = self.all_cause_mortality_rate(index)
        modeled_csmr = self.cause_specific_mortality_rate(index)
        unmodeled_csmr_raw = self._raw_unmodeled_csmr(index)
        unmodeled_csmr = self.unmodeled_csmr(index)
        cause_deleted_mortality_rate = acmr - modeled_csmr - unmodeled_csmr_raw + unmodeled_csmr
        return pd.DataFrame({'other_causes': cause_deleted_mortality_rate})

    def _get_unmodeled_csmr_source(self, index: pd.Index) -> pd.Series:
        raw_csmr = self._raw_unmodeled_csmr(index)
        paf = self.unmodeled_csmr_paf(index)
        return raw_csmr * (1 - paf)

    def _get_mortality_hazard_source(self, index: pd.Index) -> pd.Series:
        mortality_rates = pd.DataFrame(self.mortality_rate(index))
        mortality_hazard = mortality_rates.sum(axis=1)
        paf = self._mortality_hazard_paf(index)
        return mortality_hazard * (1 - paf)
