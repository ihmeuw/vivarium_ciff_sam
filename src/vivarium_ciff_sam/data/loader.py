"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""
import numpy as np
import pandas as pd

from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs import extract as vi_extract, globals as vi_globals, interface, utilities as vi_utils, utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors
from vivarium_inputs.validation.sim import validate_for_simulation

from vivarium_ciff_sam.constants import data_keys, data_values


def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.LOCATION: load_population_location,
        data_keys.POPULATION.STRUCTURE: load_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.ACMR: load_standard_data,
        data_keys.POPULATION.CRUDE_BIRTH_RATE: load_standard_data,

        data_keys.DIARRHEA.PREVALENCE: load_standard_data,
        data_keys.DIARRHEA.INCIDENCE_RATE: load_standard_data,
        data_keys.DIARRHEA.REMISSION_RATE: load_standard_data,
        data_keys.DIARRHEA.DISABILITY_WEIGHT: load_standard_data,
        data_keys.DIARRHEA.EMR: load_standard_data,
        data_keys.DIARRHEA.CSMR: load_standard_data,
        data_keys.DIARRHEA.RESTRICTIONS: load_metadata,

        data_keys.MEASLES.PREVALENCE: load_standard_data,
        data_keys.MEASLES.INCIDENCE_RATE: load_standard_data,
        data_keys.MEASLES.DISABILITY_WEIGHT: load_standard_data,
        data_keys.MEASLES.EMR: load_standard_data,
        data_keys.MEASLES.CSMR: load_standard_data,
        data_keys.MEASLES.RESTRICTIONS: load_metadata,

        data_keys.LRI.PREVALENCE: load_lri_prevalence,
        data_keys.LRI.INCIDENCE_RATE: load_standard_data,
        data_keys.LRI.REMISSION_RATE: load_standard_data,
        data_keys.LRI.DISABILITY_WEIGHT: load_standard_data,
        data_keys.LRI.EMR: load_lri_excess_mortality_rate,
        data_keys.LRI.CSMR: load_standard_data,
        data_keys.LRI.RESTRICTIONS: load_metadata,

        data_keys.WASTING.DISTRIBUTION: load_metadata,
        data_keys.WASTING.ALT_DISTRIBUTION: load_metadata,
        data_keys.WASTING.CATEGORIES: load_metadata,
        data_keys.WASTING.EXPOSURE: load_standard_data,
        data_keys.WASTING.RELATIVE_RISK: load_standard_data,
        data_keys.WASTING.PAF: load_child_wasting_paf,
    }
    return mapping[lookup_key](lookup_key, location)


def load_population_location(key: str, location: str) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f'Unrecognized key {key}')

    return location


# noinspection PyUnusedLocal
def load_population_structure(key: str, location: str) -> pd.DataFrame:
    return interface.get_population_structure(location)


# noinspection PyUnusedLocal
def load_age_bins(key: str, location: str) -> pd.DataFrame:
    all_age_bins = interface.get_age_bins().reset_index()
    return all_age_bins[all_age_bins.age_start < 5].set_index(['age_start', 'age_end', 'age_group_name'])


# noinspection PyUnusedLocal
def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    return interface.get_demographic_dimensions(location)


# noinspection PyUnusedLocal
def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    data = interface.get_measure(entity, key.measure, location).droplevel('location')
    return data


# noinspection PyUnusedLocal
def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, 'to_dict'):
        entity_metadata = entity_metadata.to_dict()
    return entity_metadata


def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = vi_utils.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = vi_utils.reshape(data)
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(data, interval_column='age', split_column_prefix='age')
    data = vi_utils.split_interval(data, interval_column='year', split_column_prefix='year')
    return vi_utils.sort_hierarchical_data(data)


def get_entity(key: str):
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]


# Project-specific data functions here

def load_lri_prevalence(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.LRI.PREVALENCE:
        incidence_rate = get_data(data_keys.LRI.INCIDENCE_RATE, location)
        prevalence = incidence_rate * data_values.LRI_DURATION / 365
        return prevalence
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_lri_excess_mortality_rate(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.LRI.EMR:
        csmr = get_data(data_keys.LRI.CSMR, location)
        prevalence = get_data(data_keys.LRI.PREVALENCE, location)
        data = (csmr / prevalence).fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
        return data
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_child_wasting_paf(key: str, location: str) -> pd.DataFrame:
    # from load_standard_data
    key = EntityKey(key)
    entity = get_entity(key)

    # from interface.get_measure
    # from vivarium_inputs.core.get_data
    location_id = utility_data.get_location_id(location) if isinstance(location, str) else location

    # from vivarium_inputs.core.get_population_attributable_fraction
    causes_map = {c.gbd_id: c for c in causes}
    data = vi_extract.extract_data(entity, 'population_attributable_fraction', location_id)

    temp = []
    # We filter paf age groups by cause level restrictions.
    for (c_id, measure), df in data.groupby(['cause_id', 'measure_id']):
        cause = causes_map[c_id]
        measure = 'yll' if measure == vi_globals.MEASURES['YLLs'] else 'yld'
        df = vi_utils.filter_data_by_restrictions(df, cause, measure, utility_data.get_age_group_ids())
        temp.append(df)
    data = pd.concat(temp, ignore_index=True)

    data = vi_utils.convert_affected_entity(data, 'cause_id')
    data['affected_measure'] = 'incidence_rate'
    data = (data.groupby(['affected_entity', 'affected_measure'])
            .apply(vi_utils.normalize, fill_value=0)
            .reset_index(drop=True))
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS
                       + ['affected_entity', 'affected_measure']
                       + vi_globals.DRAW_COLUMNS)

    # from vivarium_inputs.core.get_data
    data = vi_utils.reshape(data, value_cols=vi_globals.DRAW_COLUMNS)

    # from interface.get_measure
    data = vi_utils.scrub_gbd_conventions(data, location)
    validate_for_simulation(data, entity, key.measure, location)
    data = vi_utils.split_interval(data, interval_column='age', split_column_prefix='age')
    data = vi_utils.split_interval(data, interval_column='year', split_column_prefix='year')
    return vi_utils.sort_hierarchical_data(data).droplevel('location')
