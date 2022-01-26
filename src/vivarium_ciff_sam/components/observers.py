import itertools
from typing import Callable, Dict, Iterable, List, Tuple, Union

import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.time import Time, get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.metrics import (utilities,
                                            MortalityObserver as MortalityObserver_,
                                            DisabilityObserver as DisabilityObserver_,
                                            DiseaseObserver as DiseaseObserver_,
                                            CategoricalRiskObserver as CategoricalRiskObserver_)
from vivarium_public_health.metrics.utilities import QueryString, TransitionString

from vivarium_ciff_sam.constants import models, results, data_keys


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    def __init__(
            self,
            observer_name: str = 'False',
            by_wasting: str = 'False',
            by_sqlns: str = 'False',
            by_wasting_treatment: str = 'False',
            by_x_factor: str = 'False',
            by_stunting: str = 'False',
            by_maternal_malnutrition: str = 'False',
            by_maternal_supplementation: str = 'False',
            by_insecticide_treated_nets: str = 'False'
    ):
        self.name = f'{observer_name}_results_stratifier'
        self.by_wasting = by_wasting != 'False'
        self.by_sqlns = by_sqlns != 'False'
        self.by_wasting_treatment = by_wasting_treatment != 'False'
        self.by_x_factor = by_x_factor != 'False'
        self.by_stunting = by_stunting != 'False'
        self.by_maternal_malnutrition = by_maternal_malnutrition != 'False'
        self.by_maternal_supplementation = by_maternal_supplementation != 'False'
        self.by_insecticide_treated_nets = by_insecticide_treated_nets != 'False'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        # The only thing you should request here are resources necessary for results stratification.
        self.pipelines = {}
        columns_required = ['tracked']

        self.stratification_levels = {}

        def setup_stratification(source_name: str, is_pipeline: bool, stratification_name: str,
                                 categories: Iterable):

            def get_state_function(state: Union[str, bool, List]) -> Callable:
                return lambda pop: (pop[source_name] == state if not isinstance(state, List)
                                    else pop[source_name].isin(state))

            if type(categories) != dict:
                categories = {category: category for category in categories}

            self.stratification_levels[stratification_name] = {
                stratification_key: get_state_function(source_value)
                for stratification_key, source_value in categories.items()
            }
            if is_pipeline:
                self.pipelines[source_name] = builder.value.get_value(source_name)
            else:
                columns_required.append(data_keys.WASTING.name)

        if self.by_wasting:
            setup_stratification(data_keys.WASTING.name, False, 'wasting_state', models.WASTING.STATES)

        if self.by_stunting:
            setup_stratification(f'{data_keys.STUNTING.name}.exposure', True, 'stunting_state', range(4, 0, -1))

        if self.by_wasting_treatment:
            setup_stratification(f'{data_keys.SAM_TREATMENT.name}.exposure', True, 'sam_treatment',
                                 {'covered': data_keys.SAM_TREATMENT.COVERED_CATEGORIES,
                                  'uncovered': data_keys.SAM_TREATMENT.UNCOVERED_CATEGORIES})
            setup_stratification(f'{data_keys.MAM_TREATMENT.name}.exposure', True, 'mam_treatment',
                                 {'covered': data_keys.MAM_TREATMENT.COVERED_CATEGORIES,
                                  'uncovered': data_keys.MAM_TREATMENT.UNCOVERED_CATEGORIES})

        if self.by_sqlns:
            setup_stratification(data_keys.SQ_LNS.COVERAGE_PIPELINE, True, 'sq_lns',
                                 {'covered': True, 'uncovered': False})

        if self.by_x_factor:
            setup_stratification(
                source_name='x_factor.exposure',
                is_pipeline=True,
                stratification_name='x_factor',
                categories=results.DICHOTOMOUS_RISK_STATES
            )

        if self.by_maternal_malnutrition:
            setup_stratification(
                source_name=f'{data_keys.MATERNAL_MALNUTRITION.name}.exposure',
                is_pipeline=True,
                stratification_name='maternal_malnutrition',
                categories=results.DICHOTOMOUS_RISK_STATES
            )

        if self.by_maternal_supplementation:
            setup_stratification(
                source_name='maternal_supplementation.exposure',
                is_pipeline=True,
                stratification_name='maternal_supplementation',
                categories=results.MATERNAL_SUPPLEMENTATION_TYPES
            )

        if self.by_insecticide_treated_nets:
            setup_stratification(
                source_name=f'{data_keys.INSECTICIDE_TX_NETS.name}.exposure',
                is_pipeline=True,
                stratification_name='itn',
                categories={
                    'covered': data_keys.INSECTICIDE_TX_NETS.CAT2,
                    'uncovered': data_keys.INSECTICIDE_TX_NETS.CAT1
                }
            )

        self.population_view = builder.population.get_view(columns_required)
        self.stratification_groups: pd.Series = None

        # Ensure that the stratifier updates before its observer
        builder.event.register_listener('time_step__prepare', self.on_timestep_prepare, priority=0)

    # noinspection PyAttributeOutsideInit
    def on_timestep_prepare(self, event: Event):
        # cache stratification groups at the beginning of the time-step for use later when stratifying
        self.stratification_groups = self.get_stratification_groups(event.index)

    def get_stratification_groups(self, index: pd.Index):
        #  get values required for stratification from population view and pipelines
        pop_list = [self.population_view.get(index)] + [pd.Series(pipeline(index), name=name)
                                                        for name, pipeline in self.pipelines.items()]
        pop = pd.concat(pop_list, axis=1)

        stratification_groups = pd.Series('', index=index)
        all_stratifications = self.get_all_stratifications()
        for stratification in all_stratifications:
            stratification_group_name = '_'.join([f'{metric["metric"]}_{metric["category"]}'
                                                  for metric in stratification]).lower()
            mask = pd.Series(True, index=index)
            for metric in stratification:
                mask &= self.stratification_levels[metric['metric']][metric['category']](pop)
            stratification_groups.loc[mask] = stratification_group_name
        return stratification_groups

    def get_all_stratifications(self) -> List[Tuple[Dict[str, str], ...]]:
        """
        Gets all stratification combinations. Returns a List of Stratifications. Each Stratification is represented as a
        Tuple of Stratification Levels. Each Stratification Level is represented as a Dictionary with keys 'metric' and
        'category'. 'metric' refers to the stratification level's name, and 'category' refers to the stratification
        category.

        If no stratification levels are defined, returns a List with a single empty Tuple
        """
        # Get list of lists of metric and category pairs for each metric
        groups = [[{'metric': metric, 'category': category} for category in category_maps]
                  for metric, category_maps in self.stratification_levels.items()]
        # Get product of all stratification combinations
        return list(itertools.product(*groups))

    @staticmethod
    def get_stratification_key(stratification: Iterable[Dict[str, str]]) -> str:
        return ('' if not stratification
                else '_'.join([f'{metric["metric"]}_{metric["category"]}' for metric in stratification]))

    def group(self, pop: pd.DataFrame) -> Iterable[Tuple[Tuple[str, ...], pd.DataFrame]]:
        """Takes the full population and yields stratified subgroups.

        Parameters
        ----------
        pop
            The population to stratify.

        Yields
        ------
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        index = pop.index.intersection(self.stratification_groups.index)
        pop = pop.loc[index]
        stratification_groups = self.stratification_groups.loc[index]

        stratifications = self.get_all_stratifications()
        for stratification in stratifications:
            stratification_key = self.get_stratification_key(stratification)
            if pop.empty:
                pop_in_group = pop
            else:
                pop_in_group = pop.loc[(stratification_groups == stratification_key)]
            yield (stratification_key,), pop_in_group

    @staticmethod
    def update_labels(measure_data: Dict[str, float], labels: Tuple[str, ...]) -> Dict[str, float]:
        """Updates a dict of measure data with stratification labels.

        Parameters
        ----------
        measure_data
            The measure data with unstratified column names.
        labels
            The stratification labels. Yielded along with the population
            subgroup the measure data was produced from by a call to
            :obj:`ResultsStratifier.group`.

        Returns
        -------
            The measure data with column names updated with the stratification
            labels.

        """
        stratification_label = f'_{labels[0]}' if labels[0] else ''
        measure_data = {f'{k}{stratification_label}': v for k, v in measure_data.items()}
        return measure_data


class MortalityObserver(MortalityObserver_):

    def __init__(
            self,
            stratify_by_wasting: str = 'wasting',
            stratify_by_sq_lns: str = 'False',
            stratify_by_wasting_treatment: str = 'False',
            stratify_by_x_factor: str = 'False',
            stratify_by_stunting: str = 'False',
            stratify_by_maternal_malnutrition: str = 'False'
    ):
        super().__init__()
        self.stratifier = ResultsStratifier(
            self.name,
            stratify_by_wasting,
            stratify_by_sq_lns,
            stratify_by_wasting_treatment,
            stratify_by_x_factor,
            stratify_by_stunting,
            stratify_by_maternal_malnutrition
        )

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def metrics(self, index: pd.Index, metrics: Dict[str, float]) -> Dict[str, float]:
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        measure_getters = (
            (utilities.get_deaths, (self.causes,)),
            (utilities.get_years_of_life_lost, (self.life_expectancy, self.causes)),
        )

        for labels, pop_in_group in self.stratifier.group(pop):
            base_args = (pop_in_group, self.config.to_dict(), self.start_time, self.clock(), self.age_bins)

            for measure_getter, extra_args in measure_getters:
                measure_data = measure_getter(*base_args, *extra_args)
                measure_data = self.stratifier.update_labels(measure_data, labels)
                metrics.update(measure_data)

        # TODO remove stratification by wasting state of deaths/ylls due to PEM?

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics[results.TOTAL_YLLS_COLUMN] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)
        metrics['total_population_dead'] = len(the_dead)

        return metrics


class DisabilityObserver(DisabilityObserver_):

    def __init__(
            self,
            stratify_by_wasting: str = 'wasting',
            stratify_by_sq_lns: str = 'False',
            stratify_by_wasting_treatment: str = 'False',
            stratify_by_x_factor: str = 'False',
            stratify_by_stunting: str = 'False',
            stratify_by_maternal_malnutrition: str = 'False'
    ):
        super().__init__()
        self.stratifier = ResultsStratifier(
            self.name,
            stratify_by_wasting,
            stratify_by_sq_lns,
            stratify_by_wasting_treatment,
            stratify_by_x_factor,
            stratify_by_stunting,
            stratify_by_maternal_malnutrition
        )

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        self.update_metrics(pop)

        pop.loc[:, results.TOTAL_YLDS_COLUMN] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    def update_metrics(self, pop: pd.DataFrame):
        for labels, pop_in_group in self.stratifier.group(pop):
            base_args = (pop_in_group, self.config.to_dict(), self.clock().year, self.step_size(), self.age_bins,
                         self.disability_weight_pipelines, self.causes)
            measure_data = self.stratifier.update_labels(utilities.get_years_lived_with_disability(*base_args), labels)
            self.years_lived_with_disability.update(measure_data)

        # TODO remove stratification by wasting state of ylds due to PEM?


class DiseaseObserver(DiseaseObserver_):

    def __init__(
            self,
            disease: str,
            stratify_by_wasting: str = 'wasting',
            stratify_by_sq_lns: str = 'False',
            stratify_by_wasting_treatment: str = 'False',
            stratify_by_x_factor: str = 'False',
            stratify_by_stunting: str = 'False',
            stratify_by_maternal_malnutrition: str = 'False'
    ):
        super().__init__(disease)
        self.stratifier = ResultsStratifier(
            self.name,
            stratify_by_wasting,
            stratify_by_sq_lns,
            stratify_by_wasting_treatment,
            stratify_by_x_factor,
            stratify_by_stunting,
            stratify_by_maternal_malnutrition
        )

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    @property
    def name(self):
        return f'{self.disease}_disease_observer'

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        for labels, pop_in_group in self.stratifier.group(pop):
            for state in self.states:
                # noinspection PyTypeChecker
                state_person_time_this_step = utilities.get_state_person_time(
                    pop_in_group, self.config, self.disease, state, self.clock().year, event.step_size, self.age_bins
                )
                state_person_time_this_step = self.stratifier.update_labels(state_person_time_this_step, labels)
                self.person_time.update(state_person_time_this_step)

        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    def on_collect_metrics(self, event: Event):
        pop = self.population_view.get(event.index)
        for labels, pop_in_group in self.stratifier.group(pop):
            for transition in self.transitions:
                transition = TransitionString(transition)
                # noinspection PyTypeChecker
                transition_counts_this_step = utilities.get_transition_count(pop_in_group, self.config, self.disease,
                                                                             transition, event.time, self.age_bins)
                transition_counts_this_step = self.stratifier.update_labels(transition_counts_this_step, labels)
                self.counts.update(transition_counts_this_step)

    def __repr__(self) -> str:
        return f"DiseaseObserver({self.disease})"

    ################
    # Name Getters #
    ################

    @staticmethod
    def get_previous_state_column_name(disease_name: str) -> str:
        return f'previous_{disease_name}'


class CategoricalRiskObserver(CategoricalRiskObserver_):

    def __init__(
            self,
            risk: str,
            stratify_by_wasting: str = 'False',
            stratify_by_sq_lns: str = 'False',
            stratify_by_wasting_treatment: str = 'False',
            stratify_by_x_factor: str = 'False',
            stratify_by_stunting: str = 'False',
            stratify_by_maternal_malnutrition: str = 'False'
    ):
        super().__init__(risk)
        self.stratifier = ResultsStratifier(
            self.name,
            stratify_by_wasting,
            stratify_by_sq_lns,
            stratify_by_wasting_treatment,
            stratify_by_x_factor,
            stratify_by_stunting,
            stratify_by_maternal_malnutrition
        )

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def on_time_step_prepare(self, event: Event):
        pop = pd.concat([self.population_view.get(event.index), pd.Series(self.exposure(event.index), name=self.risk)],
                        axis=1)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        for labels, pop_in_group in self.stratifier.group(pop):
            for category in self.categories:
                # noinspection PyTypeChecker
                state_person_time_this_step = utilities.get_state_person_time(
                    pop_in_group, self.config, self.risk, category, self.clock().year, event.step_size, self.age_bins
                )
                state_person_time_this_step = self.stratifier.update_labels(state_person_time_this_step, labels)
                self.person_time.update(state_person_time_this_step)


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
            stratify_by_wasting: str = 'False',
            stratify_by_sq_lns: str = 'False',
            stratify_by_wasting_treatment: str = 'False',
            stratify_by_x_factor: str = 'False',
            stratify_by_stunting: str = 'False',
            stratify_by_maternal_malnutrition: str = 'True',
            stratify_by_maternal_supplementation: str = 'True',
            stratify_by_insecticide_treated_nets: str = 'True'
    ):
        self.configuration_defaults = self._get_configuration_defaults()
        self.stratifier = ResultsStratifier(
            self.name,
            stratify_by_wasting,
            stratify_by_sq_lns,
            stratify_by_wasting_treatment,
            stratify_by_x_factor,
            stratify_by_stunting,
            stratify_by_maternal_malnutrition,
            stratify_by_maternal_supplementation,
            stratify_by_insecticide_treated_nets
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
