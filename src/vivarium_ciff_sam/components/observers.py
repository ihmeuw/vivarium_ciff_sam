from typing import Callable, Dict, List, Tuple

import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView
from vivarium.framework.time import Time, get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.metrics.stratification import (
    ResultsStratifier as ResultsStratifier_,
    Source,
    SourceType,
)

from vivarium_ciff_sam.constants import models, results, data_keys


class ResultsStratifier(ResultsStratifier_):
    """Centralized component for handling results stratification.

    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    def register_stratifications(self, builder: Builder) -> None:
        """Register each desired stratification with calls to _setup_stratification"""
        super().register_stratifications(builder)

        self.setup_stratification(
            builder,
            name="wasting_state",
            sources=[Source(data_keys.WASTING.name, SourceType.COLUMN)],
            categories=set(models.WASTING.STATES),
        )

        self.setup_stratification(
            builder,
            name="sam_treatment",
            sources=[Source(f'{data_keys.SAM_TREATMENT.name}.exposure', SourceType.PIPELINE)],
            categories=ResultsStratifier.TREATMENT_CATEGORIES,
            mapper=self.wasting_treatment_stratification_mapper,
        )

        self.setup_stratification(
            builder,
            name="mam_treatment",
            sources=[Source(f'{data_keys.MAM_TREATMENT.name}.exposure', SourceType.PIPELINE)],
            categories=ResultsStratifier.TREATMENT_CATEGORIES,
            mapper=self.wasting_treatment_stratification_mapper,
        )

        self.setup_stratification(
            builder,
            name="sqlns",
            sources=[Source(data_keys.SQ_LNS.COVERAGE_PIPELINE, SourceType.PIPELINE)],
            categories=ResultsStratifier.TREATMENT_CATEGORIES,
            mapper=self.sqlns_stratification_mapper,
        )

        self.setup_stratification(
            builder,
            name="therapeutic_zinc",
            sources=[Source('therapeutic_zinc.exposure', SourceType.PIPELINE)],
            categories=ResultsStratifier.TREATMENT_CATEGORIES,
            mapper=self.treatment_stratification_mapper,
        )

        self.setup_stratification(
            builder,
            name="preventative_zinc",
            sources=[Source('preventative_zinc.exposure', SourceType.PIPELINE)],
            categories=ResultsStratifier.TREATMENT_CATEGORIES,
            mapper=self.treatment_stratification_mapper,
        )

        self.setup_stratification(
            builder,
            name="diarrhea",
            sources=[Source(data_keys.DIARRHEA.name, SourceType.COLUMN)],
            categories=set(models.DIARRHEA.STATES),
        )

    ###########################
    # Stratifications Details #
    ###########################

    TREATMENT_CATEGORIES = {"covered", "uncovered"}

    # noinspection PyMethodMayBeStatic
    def wasting_treatment_stratification_mapper(self, row: pd.Series) -> str:
        return {
            "cat3": "covered",
            "cat2": "covered",
            "cat1": "uncovered",
        }[row.squeeze()]

    # noinspection PyMethodMayBeStatic
    def sqlns_stratification_mapper(self, row: pd.Series) -> str:
        return "covered" if row.squeeze() else "uncovered"

    # noinspection PyMethodMayBeStatic
    def treatment_stratification_mapper(self, row: pd.Series) -> str:
        return {
            "cat2": "covered",
            "cat1": "uncovered",
        }[row.squeeze()]


class BirthObserver:
    configuration_defaults = {
        'metrics': {
            'birth': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(
            self,
            stratify_by_maternal_malnutrition: str = 'maternal_malnutrition',
            stratify_by_maternal_supplementation: str = 'maternal_supplementation',
            stratify_by_insecticide_treated_nets: str = 'insecticide_treated_nets'
    ):
        self.configuration_defaults = self._get_configuration_defaults()
        self.stratifier = ResultsStratifier(
            self.name,
            by_maternal_malnutrition=stratify_by_maternal_malnutrition,
            by_maternal_supplementation=stratify_by_maternal_supplementation,
            by_insecticide_treated_nets=stratify_by_insecticide_treated_nets
        )

        self.tracked_column_name = 'tracked'
        self.entrance_time_column_name = 'entrance_time'

        self.birth_weight_pipeline_name = 'low_birth_weight.exposure'
        self.metrics_pipeline_name = 'metrics'

    def __repr__(self):
        return 'BirthObserver()'

    ##########################
    # Initialization methods #
    ##########################

    # noinspection PyMethodMayBeStatic
    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            'metrics': {
                f'birth': BirthObserver.configuration_defaults['metrics']['birth']
            }
        }

    ##############
    # Properties #
    ##############

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    @property
    def name(self):
        return 'birth_observer'

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = self._get_clock(builder)
        self.configuration = self._get_configuration(builder)
        self.start_time = self._get_start_time(builder)
        self.age_bins = self._get_age_bins(builder)
        self.pipelines = self._get_pipelines(builder)
        self.population_view = self._get_population_view(builder)

        self._register_metrics_modifier(builder)

    # noinspection PyMethodMayBeStatic
    def _get_clock(self, builder: Builder) -> Callable[[], Time]:
        return builder.time.clock()

    # noinspection PyMethodMayBeStatic
    def _get_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.metrics.birth

    # noinspection PyMethodMayBeStatic
    def _get_start_time(self, builder: Builder) -> pd.Timestamp:
        return get_time_stamp(builder.configuration.time.start)

    # noinspection PyMethodMayBeStatic
    def _get_age_bins(self, builder: Builder) -> pd.DataFrame:
        return utilities.get_age_bins(builder)

    def _get_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {
            self.birth_weight_pipeline_name: builder.value.get_value(self.birth_weight_pipeline_name),
        }

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(['sex', self.tracked_column_name, self.entrance_time_column_name])

    def _register_metrics_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.metrics_pipeline_name,
            self._metrics,
            requires_columns=[self.entrance_time_column_name],
            requires_values=[self.birth_weight_pipeline_name]
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    # noinspection PyUnusedLocal
    def _metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        pipelines = [
            pd.Series(pipeline(index), name=pipeline_name)
            for pipeline_name, pipeline in self.pipelines.items()
        ]
        pop = pd.concat([self.population_view.get(index)] + pipelines, axis=1)

        measure_getters = (
            (self._get_births, ()),
            (self._get_birth_weight_sum, ()),
            (self._get_births, (results.LOW_BIRTH_WEIGHT_CUTOFF,)),
        )

        config_dict = self.configuration.to_dict()
        base_filter = QueryString(f'"{{start_time}}" <= {self.entrance_time_column_name} '
                                  f'and {self.entrance_time_column_name} < "{{end_time}}"')
        time_spans = utilities.get_time_iterable(config_dict, self.start_time, self.clock())

        for labels, pop_in_group in self.stratifier.group(pop):
            args = (pop_in_group, base_filter, self.configuration.to_dict(), time_spans, self.age_bins)

            for measure_getter, extra_args in measure_getters:
                measure_data = measure_getter(*args, *extra_args)
                measure_data = self.stratifier.update_labels(measure_data, labels)
                metrics.update(measure_data)

        return metrics

    def _get_births(self, pop: pd.DataFrame, base_filter: QueryString, configuration: Dict,
                    time_spans: List[Tuple[str, Tuple[pd.Timestamp, pd.Timestamp]]], age_bins: pd.DataFrame,
                    cutoff_weight: float = None) -> Dict[str, float]:
        if cutoff_weight:
            base_filter += (
                QueryString('{column} <= {cutoff}')
                .format(column=f'`{self.birth_weight_pipeline_name}`', cutoff=cutoff_weight)
            )
            measure = 'low_weight_births'
        else:
            measure = 'total_births'

        base_key = utilities.get_output_template(**configuration).substitute(measure=measure)

        births = {}
        for year, (year_start, year_end) in time_spans:
            year_filter = base_filter.format(start_time=year_start, end_time=year_end)
            year_key = base_key.substitute(year=year)
            group_births = utilities.get_group_counts(pop, year_filter, year_key, configuration, age_bins)
            births.update(group_births)
        return births

    def _get_birth_weight_sum(self, pop: pd.DataFrame, base_filter: QueryString, configuration: Dict,
                              time_spans: List[Tuple[str, Tuple[pd.Timestamp, pd.Timestamp]]],
                              age_bins: pd.DataFrame) -> Dict[str, float]:

        base_key = utilities.get_output_template(**configuration).substitute(measure='birth_weight_sum')

        birth_weight_sum = {}
        for year, (year_start, year_end) in time_spans:
            year_filter = base_filter.format(start_time=year_start, end_time=year_end)
            year_key = base_key.substitute(year=year)
            group_birth_weight_sums = utilities.get_group_counts(pop, year_filter, year_key, configuration, age_bins,
                                                                 lambda df: df[self.birth_weight_pipeline_name].sum())
            birth_weight_sum.update(group_birth_weight_sums)
        return birth_weight_sum
