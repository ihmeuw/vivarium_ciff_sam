import itertools
from typing import Callable, Dict, Iterable, List, Tuple, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_public_health.metrics import (MortalityObserver as MortalityObserver_,
                                            DisabilityObserver as DisabilityObserver_,
                                            DiseaseObserver as DiseaseObserver_,
                                            CategoricalRiskObserver as CategoricalRiskObserver_)
from vivarium_public_health.metrics.utilities import (get_deaths, get_state_person_time, get_transition_count,
                                                      get_years_lived_with_disability, get_years_of_life_lost,
                                                      TransitionString)

from vivarium_ciff_sam.constants import models, results, data_keys


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    def __init__(self, observer_name: str, by_wasting: str, by_sqlns: str, by_wasting_treatment: str, by_stunting: str):
        self.name = f'{observer_name}_results_stratifier'
        self.by_wasting = by_wasting != 'False'
        self.by_sqlns = by_sqlns != 'False'
        self.by_wasting_treatment = by_wasting_treatment != 'False'
        self.by_stunting = by_stunting != 'False'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        # The only thing you should request here are resources necessary for results stratification.
        self.pipelines = {}
        columns_required = ['tracked']

        def get_state_function(column_name: str, state: Union[str, bool]) -> Callable:
            return lambda pop: pop[column_name] == state

        self.stratification_levels = {}

        if self.by_wasting:
            self.stratification_levels['wasting_state'] = {
                wasting_state: get_state_function(data_keys.WASTING.name, wasting_state)
                for wasting_state in models.WASTING.STATES
            }
            columns_required.append(data_keys.WASTING.name)

        if self.by_stunting:
            self.stratification_levels['stunting_state'] = {
                f'cat{i}': get_state_function(data_keys.STUNTING.name, f'cat{i}')
                for i in range(4, 0, -1)
            }
            self.pipelines[data_keys.STUNTING.name] = builder.value.get_value('child_stunting.exposure')

        if self.by_wasting_treatment:
            wasting_treatment_key = data_keys.WASTING_NON_TREATMENT.name
            self.stratification_levels['wasting_treatment'] = {
                coverage: get_state_function(wasting_treatment_key, category)
                for category, coverage in (('cat2', 'covered'), ('cat1', 'uncovered'))
            }
            self.pipelines[wasting_treatment_key] = builder.value.get_value('wasting_non_treatment.exposure')

        if self.by_sqlns:
            self.stratification_levels['sq_lns'] = {
                coverage: get_state_function(data_keys.SQ_LNS.name, coverage == 'covered')
                for coverage in ['covered', 'uncovered']
            }
            self.pipelines[data_keys.SQ_LNS.name] = builder.value.get_value(data_keys.SQ_LNS.COVERAGE)

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

    def __init__(self, stratify_by_wasting: str = 'wasting', stratify_by_sq_lns: str = 'False',
                 stratify_by_wasting_treatment: str = 'False', stratify_by_stunting: str = 'False'):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name, stratify_by_wasting, stratify_by_sq_lns,
                                            stratify_by_wasting_treatment, stratify_by_stunting)

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def metrics(self, index: pd.Index, metrics: Dict[str, float]) -> Dict[str, float]:
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        measure_getters = (
            (get_deaths, (self.causes,)),
            (get_years_of_life_lost, (self.life_expectancy, self.causes)),
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

    def __init__(self, stratify_by_wasting: str = 'wasting', stratify_by_sq_lns: str = 'False',
                 stratify_by_wasting_treatment: str = 'False', stratify_by_stunting: str = 'False'):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name, stratify_by_wasting, stratify_by_sq_lns,
                                            stratify_by_wasting_treatment, stratify_by_stunting)

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
            measure_data = self.stratifier.update_labels(get_years_lived_with_disability(*base_args), labels)
            self.years_lived_with_disability.update(measure_data)

        # TODO remove stratification by wasting state of ylds due to PEM?


class DiseaseObserver(DiseaseObserver_):

    def __init__(self, disease: str, stratify_by_wasting: str = 'wasting', stratify_by_sq_lns: str = 'False',
                 stratify_by_wasting_treatment: str = 'False', stratify_by_stunting: str = 'False'):
        super().__init__(disease)
        self.stratifier = ResultsStratifier(self.name, stratify_by_wasting, stratify_by_sq_lns,
                                            stratify_by_wasting_treatment, stratify_by_stunting)

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
                state_person_time_this_step = get_state_person_time(pop_in_group, self.config, self.disease, state,
                                                                    self.clock().year, event.step_size, self.age_bins)
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
                transition_counts_this_step = get_transition_count(pop_in_group, self.config, self.disease, transition,
                                                                   event.time, self.age_bins)
                transition_counts_this_step = self.stratifier.update_labels(transition_counts_this_step, labels)
                self.counts.update(transition_counts_this_step)

    def __repr__(self) -> str:
        return f"DiseaseObserver({self.disease})"


class CategoricalRiskObserver(CategoricalRiskObserver_):

    def __init__(self, risk: str, stratify_by_wasting: str = 'False', stratify_by_sq_lns: str = 'False',
                 stratify_by_wasting_treatment: str = 'False', stratify_by_stunting: str = 'False'):
        super().__init__(risk)
        self.stratifier = ResultsStratifier(self.name, stratify_by_wasting, stratify_by_sq_lns,
                                            stratify_by_wasting_treatment, stratify_by_stunting)

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
                state_person_time_this_step = get_state_person_time(pop_in_group, self.config, self.risk, category,
                                                                    self.clock().year, event.step_size, self.age_bins)
                state_person_time_this_step = self.stratifier.update_labels(state_person_time_this_step, labels)
                self.person_time.update(state_person_time_this_step)
