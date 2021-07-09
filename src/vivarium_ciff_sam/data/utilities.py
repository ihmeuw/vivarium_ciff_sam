from itertools import product
from numbers import Real
from typing import List
import warnings

import pandas as pd

from gbd_mapping import causes, covariates, risk_factors, Cause, ModelableEntity, RiskFactor
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import constants as gbd_constants, gbd
from vivarium_gbd_access.utilities import get_draws, query
from vivarium_inputs import globals as vi_globals, utilities as vi_utils, utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors
from vivarium_inputs.validation.raw import check_metadata
from vivarium_inputs.validation.sim import validate_for_simulation

from vivarium_ciff_sam.constants.metadata import GBD_2020_AGE_GROUPS, GBD_2020_ROUND_ID


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


def get_entity(key: EntityKey) -> ModelableEntity:
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors
    }
    return type_map[key.type][key.name]


def get_gbd_2020_entity(key: str) -> ModelableEntity:
    # from load_standard_data
    entity = get_entity(key)

    if isinstance(entity, RiskFactor) or isinstance(entity, Cause):
        # Set risk factor age restrictions for GBD 2020
        if 'yll_age_group_id_start' in entity.restrictions:
            entity.restrictions.yll_age_group_id_start = min(GBD_2020_AGE_GROUPS)
        if 'yld_age_group_id_start' in entity.restrictions:
            entity.restrictions.yld_age_group_id_start = min(GBD_2020_AGE_GROUPS)
        if 'yll_age_group_id_end' in entity.restrictions:
            entity.restrictions.yll_age_group_id_end = max(GBD_2020_AGE_GROUPS)
        if 'yld_age_group_id_end' in entity.restrictions:
            entity.restrictions.yld_age_group_id_end = max(GBD_2020_AGE_GROUPS)

    return entity


def get_child_wasting_data(key: EntityKey, entity: ModelableEntity, location: str, source: str) -> pd.DataFrame:
    # from interface.get_measure
    # from vivarium_inputs.core.get_data
    location_id = utility_data.get_location_id(location) if isinstance(location, str) else location

    # from vivarium_inputs.core.get_{measure}
    # from vivarium_inputs.extract.extract_data
    check_metadata(entity, key.measure)

    # from vivarium_inputs.extract.extract_{measure}
    # from vivarium_gbd_access.gbd.get_{measure}
    data = get_draws(gbd_id_type='rei_id',
                     gbd_id=entity.gbd_id,
                     source=source,
                     location_id=location_id,
                     sex_id=gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE,
                     age_group_id=GBD_2020_AGE_GROUPS,
                     gbd_round_id=GBD_2020_ROUND_ID,
                     decomp_step='iterative',
                     status='best')
    return data


def validate_and_reshape_child_wasting_data(data: pd.DataFrame, entity: ModelableEntity, key: EntityKey,
                                            location: str) -> pd.DataFrame:
    # from vivarium_inputs.core.get_data
    data = vi_utils.reshape(data, value_cols=vi_globals.DRAW_COLUMNS)

    # from interface.get_measure
    data = _scrub_gbd_2020_conventions(data, location)

    estimation_years = _get_gbd_2020_estimation_years()
    validation_years = pd.DataFrame({'year_start': range(min(estimation_years), max(estimation_years) + 1)})
    validation_years['year_end'] = validation_years['year_start'] + 1

    validate_for_simulation(data, entity, key.measure, location, years=validation_years,
                            age_bins=_get_gbd_2020_age_bins())
    data = vi_utils.split_interval(data, interval_column='age', split_column_prefix='age')
    data = vi_utils.split_interval(data, interval_column='year', split_column_prefix='year')
    data = vi_utils.sort_hierarchical_data(data).droplevel('location')
    return data


def normalize_gbd_2020(data: pd.DataFrame, fill_value: Real = None,
                       cols_to_fill: List[str] = vi_globals.DRAW_COLUMNS) -> pd.DataFrame:
    data = vi_utils.normalize_sex(data, fill_value, cols_to_fill)

    # vi_inputs.normalize_year(data)
    binned_years = _get_gbd_2020_estimation_years()   # get GBD 2020 estimation years
    years = {'annual': list(range(min(binned_years), max(binned_years) + 1)), 'binned': binned_years}

    if 'year_id' not in data:
        # Data doesn't vary by year, so copy for each year.
        df = []
        for year in years['annual']:
            fill_data = data.copy()
            fill_data['year_id'] = year
            df.append(fill_data)
        data = pd.concat(df, ignore_index=True)
    elif set(data.year_id) == set(years['binned']):
        data = vi_utils.interpolate_year(data)
    else:  # set(data.year_id.unique()) == years['annual']
        pass

    # Dump extra data.
    data = data[data.year_id.isin(years['annual'])]

    data = vi_utils.normalize_age(data, fill_value, cols_to_fill)
    return data


def _get_gbd_2020_estimation_years() -> List[int]:
    """Gets the estimation years for a particular gbd round."""
    from db_queries import get_demographics
    warnings.filterwarnings("default", module="db_queries")

    return get_demographics(gbd_constants.CONN_DEFS.EPI, gbd_round_id=GBD_2020_ROUND_ID)['year_id']


def _scrub_gbd_2020_conventions(data: pd.DataFrame, location: str) -> pd.DataFrame:
    data = vi_utils.scrub_location(data, location)
    data = vi_utils.scrub_sex(data)
    data = _scrub_gbd_2020_age(data)
    data = vi_utils.scrub_year(data)
    data = vi_utils.scrub_affected_entity(data)
    return data


def _scrub_gbd_2020_age(data: pd.DataFrame) -> pd.DataFrame:
    if 'age_group_id' in data.index.names:
        age_bins = _get_gbd_2020_age_bins().set_index('age_group_id')
        id_levels = data.index.levels[data.index.names.index('age_group_id')]
        interval_levels = [pd.Interval(age_bins.age_start[age_id], age_bins.age_end[age_id], closed='left')
                           for age_id in id_levels]
        data.index = data.index.rename('age', 'age_group_id').set_levels(interval_levels, 'age')
    return data


def _get_gbd_2020_age_bins() -> pd.DataFrame:
    # from gbd.get_age_bins()
    q = f"""
                SELECT age_group_id,
                       age_group_years_start,
                       age_group_years_end,
                       age_group_name
                FROM age_group
                WHERE age_group_id IN ({','.join([str(a) for a in GBD_2020_AGE_GROUPS])})
                """
    raw_age_bins = query(q, 'shared')

    # from utility_data.get_age_bins()
    age_bins = (
        raw_age_bins[['age_group_id', 'age_group_name', 'age_group_years_start', 'age_group_years_end']]
        .rename(columns={'age_group_years_start': 'age_start', 'age_group_years_end': 'age_end'})
    )
    return age_bins


def filter_relative_risk_to_cause_restrictions(data: pd.DataFrame) -> pd.DataFrame:
    """ It applies age restrictions according to affected causes
    and affected measures. If affected measure is incidence_rate,
    it applies the yld_age_restrictions. If affected measure is
    excess_mortality_rate, it applies the yll_age_restrictions to filter
    the relative_risk data"""

    temp = []
    affected_entities = set(data.affected_entity)
    affected_measures = set(data.affected_measure)
    for cause, measure in product(affected_entities, affected_measures):
        df = data[(data.affected_entity == cause) & (data.affected_measure == measure)]
        cause = get_gbd_2020_entity(EntityKey(f'cause.{cause}.{measure}'))
        if measure == 'excess_mortality_rate':
            start, end = vi_utils.get_age_group_ids_by_restriction(cause, 'yll')
        else:  # incidence_rate
            start, end = vi_utils.get_age_group_ids_by_restriction(cause, 'yld')
        temp.append(df[df.age_group_id.isin(range(start, end + 1))])
    data = pd.concat(temp)
    return data
