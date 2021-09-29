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

from gbd_mapping import sequelae
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import constants as gbd_constants
from vivarium_inputs import globals as vi_globals, interface, utilities as vi_utils, utility_data

from vivarium_ciff_sam.constants import data_keys, data_values
from vivarium_ciff_sam.data import utilities

from vivarium_ciff_sam.utilities import get_random_variable_draws


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
        data_keys.POPULATION.ACMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.POPULATION.CRUDE_BIRTH_RATE: load_standard_data,

        data_keys.DIARRHEA.PREVALENCE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.DIARRHEA.INCIDENCE_RATE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.DIARRHEA.REMISSION_RATE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.DIARRHEA.DISABILITY_WEIGHT: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.DIARRHEA.EMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.DIARRHEA.CSMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.DIARRHEA.RESTRICTIONS: load_metadata,

        data_keys.MEASLES.PREVALENCE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.MEASLES.INCIDENCE_RATE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.MEASLES.DISABILITY_WEIGHT: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.MEASLES.EMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.MEASLES.CSMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.MEASLES.RESTRICTIONS: load_metadata,

        data_keys.LRI.PREVALENCE: load_lri_prevalence,
        data_keys.LRI.INCIDENCE_RATE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.REMISSION_RATE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.DISABILITY_WEIGHT: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.EMR: load_lri_excess_mortality_rate,
        data_keys.LRI.CSMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.RESTRICTIONS: load_metadata,

        data_keys.PEM.MAM_DISABILITY_WEIGHT: load_pem_disability_weight,
        data_keys.PEM.SAM_DISABILITY_WEIGHT: load_pem_disability_weight,
        data_keys.PEM.EMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.PEM.CSMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.PEM.RESTRICTIONS: load_metadata,

        data_keys.WASTING.DISTRIBUTION: load_metadata,
        data_keys.WASTING.ALT_DISTRIBUTION: load_metadata,
        data_keys.WASTING.CATEGORIES: load_metadata,
        data_keys.WASTING.EXPOSURE: load_gbd_2020_exposure,
        data_keys.WASTING.RELATIVE_RISK: load_gbd_2020_rr,
        data_keys.WASTING.PAF: load_paf,

        data_keys.STUNTING.DISTRIBUTION: load_metadata,
        data_keys.STUNTING.ALT_DISTRIBUTION: load_metadata,
        data_keys.STUNTING.CATEGORIES: load_metadata,
        data_keys.STUNTING.EXPOSURE: load_gbd_2020_exposure,
        data_keys.STUNTING.RELATIVE_RISK: load_gbd_2020_rr,
        data_keys.STUNTING.PAF: load_paf,

        data_keys.WASTING_TREATMENT.DISTRIBUTION: load_wasting_treatment_distribution,
        data_keys.WASTING_TREATMENT.CATEGORIES: load_wasting_treatment_categories,
        data_keys.WASTING_TREATMENT.EXPOSURE: load_wasting_treatment_exposure,
        data_keys.WASTING_TREATMENT.RELATIVE_RISK: load_wasting_treatment_rr,
        data_keys.WASTING_TREATMENT.PAF: load_paf,

        data_keys.X_FACTOR.DISTRIBUTION: load_low_maternal_bmi_distribution,
        data_keys.X_FACTOR.CATEGORIES: load_low_maternal_bmi_categories,
        data_keys.X_FACTOR.EXPOSURE: load_low_maternal_bmi_exposure,
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
    all_age_bins = (
        utilities.get_gbd_2020_age_bins()
        .set_index(['age_start', 'age_end', 'age_group_name'])
        .sort_index()
    )
    return all_age_bins


# noinspection PyUnusedLocal
def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.POPULATION.DEMOGRAPHY:
        return utilities.get_gbd_2020_demographic_dimensions()
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    data = interface.get_measure(entity, key.measure, location).droplevel('location')
    return data


# noinspection PyUnusedLocal
def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, 'to_dict'):
        entity_metadata = entity_metadata.to_dict()
    return entity_metadata


# Project-specific data functions here

def load_standard_gbd_2019_data_as_gbd_2020_data(key: str, location: str) -> pd.DataFrame:
    gbd_2019_data = load_standard_data(key, location)
    return utilities.reshape_gbd_2019_data_as_gbd_2020_data(gbd_2019_data)


def load_lri_prevalence(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.LRI.PREVALENCE:
        incidence_rate = get_data(data_keys.LRI.INCIDENCE_RATE, location)
        early_neonatal_prevalence = (incidence_rate[incidence_rate.index.get_level_values('age_start') == 0.0]
                                     * data_values.EARLY_NEONATAL_CAUSE_DURATION / 365)
        all_other_prevalence = (incidence_rate[incidence_rate.index.get_level_values('age_start') != 0.0]
                                * data_values.LRI_DURATION / 365)
        prevalence = pd.concat([early_neonatal_prevalence, all_other_prevalence])
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


def load_gbd_2020_exposure(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_gbd_2020_entity(key)

    data = utilities.get_gbd_2020_data(key, entity, location, gbd_constants.SOURCES.EXPOSURE, 'rei_id')
    data['rei_id'] = entity.gbd_id

    # from vivarium_inputs.extract.extract_exposure
    allowable_measures = [vi_globals.MEASURES['Proportion'], vi_globals.MEASURES['Continuous'],
                          vi_globals.MEASURES['Prevalence']]
    proper_measure_id = set(data.measure_id).intersection(allowable_measures)
    if len(proper_measure_id) != 1:
        raise vi_globals.DataAbnormalError(f'Exposure data have {len(proper_measure_id)} measure id(s). '
                                           f'Data should have exactly one id out of {allowable_measures} '
                                           f'but came back with {proper_measure_id}.')
    data = data[data.measure_id == proper_measure_id.pop()]

    # from vivarium_inputs.core.get_exposure
    data = data.drop('modelable_entity_id', 'columns')
    tmrel_cat = utility_data.get_tmrel_category(entity)
    unexposed = data[data.parameter == tmrel_cat]
    exposed = [data[data.parameter == cat] for cat in [data_keys.WASTING.CAT3, data_keys.WASTING.CAT2,
                                                       data_keys.WASTING.CAT1]]
    #  FIXME: We fill 1 as exposure of tmrel category, which is not correct.
    data = pd.concat([utilities.normalize_gbd_2020(cat, fill_value=0) for cat in exposed]
                     + [utilities.normalize_gbd_2020(unexposed, fill_value=1)],
                     ignore_index=True)

    # normalize so all categories sum to 1
    cols = list(set(data.columns).difference(vi_globals.DRAW_COLUMNS + ['parameter']))
    data = data.set_index(cols + ['parameter'])
    sums = (
        data.groupby(cols)[vi_globals.DRAW_COLUMNS].sum()
            .reindex(index=data.index)
    )
    data = data.divide(sums).reset_index()

    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS + ['parameter'])
    data = utilities.validate_and_reshape_child_wasting_data(data, entity, key, location)

    if key == data_keys.STUNTING.EXPOSURE:
        # Remove neonatal exposure
        neonatal_age_ends = data.index.get_level_values('age_end').unique()[:2]
        data.loc[data.index.get_level_values('age_end').isin(neonatal_age_ends)] = 0.0
        data.loc[data.index.get_level_values('age_end').isin(neonatal_age_ends)
                 & (data.index.get_level_values('parameter') == data_keys.STUNTING.CAT4)] = 1.0
    return data


def load_gbd_2020_rr(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_gbd_2020_entity(key)

    data = utilities.get_gbd_2020_data(key, entity, location, gbd_constants.SOURCES.RR, 'rei_id')
    data['rei_id'] = entity.gbd_id

    # from vivarium_inputs.extract.extract_relative_risk
    data = vi_utils.filter_to_most_detailed_causes(data)

    # from vivarium_inputs.core.get_relative_risk
    data = vi_utils.convert_affected_entity(data, 'cause_id')
    morbidity = data.morbidity == 1
    mortality = data.mortality == 1
    data.loc[morbidity & mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[morbidity & ~mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[~morbidity & mortality, 'affected_measure'] = 'excess_mortality_rate'
    data = utilities.filter_relative_risk_to_cause_restrictions(data)

    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + ['affected_entity', 'affected_measure', 'parameter']
                       + vi_globals.DRAW_COLUMNS)
    data = (data.groupby(['affected_entity', 'parameter'])
            .apply(utilities.normalize_gbd_2020, fill_value=1)
            .reset_index(drop=True))

    tmrel_cat = utility_data.get_tmrel_category(entity)
    tmrel_mask = data.parameter == tmrel_cat
    data.loc[tmrel_mask, vi_globals.DRAW_COLUMNS] = (data.loc[tmrel_mask, vi_globals.DRAW_COLUMNS]
                                                     .mask(np.isclose(data.loc[tmrel_mask, vi_globals.DRAW_COLUMNS],
                                                                      1.0), 1.0))

    data = utilities.validate_and_reshape_child_wasting_data(data, entity, key, location)

    if key in [data_keys.WASTING.RELATIVE_RISK, data_keys.STUNTING.RELATIVE_RISK]:
        # Remove neonatal relative risks
        neonatal_age_ends = data.index.get_level_values('age_end').unique()[:2]
        data.loc[data.index.get_level_values('age_end').isin(neonatal_age_ends)] = 1.0

    return data


def load_paf(key: str, location: str) -> pd.DataFrame:
    if key in [data_keys.WASTING.PAF, data_keys.STUNTING.PAF, data_keys.WASTING_TREATMENT.PAF]:
        risk = {
            data_keys.WASTING.PAF: data_keys.WASTING,
            data_keys.STUNTING.PAF: data_keys.STUNTING,
            data_keys.WASTING_TREATMENT.PAF: data_keys.WASTING_TREATMENT
        }[key]

        exp = get_data(risk.EXPOSURE, location)
        rr = get_data(risk.RELATIVE_RISK, location)

        # paf = (sum_categories(exp * rr) - 1) / sum_categories(exp * rr)
        sum_exp_x_rr = (
            (exp * rr)
            .groupby(list(set(rr.index.names) - {'parameter'})).sum()
            .reset_index()
            .set_index(rr.index.names[:-1])
        )
        paf = (sum_exp_x_rr - 1) / sum_exp_x_rr
        return paf
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_pem_disability_weight(key: str, location: str) -> pd.DataFrame:
    if key in [data_keys.PEM.MAM_DISABILITY_WEIGHT, data_keys.PEM.SAM_DISABILITY_WEIGHT]:
        key_sequelae_map = {
            data_keys.PEM.MAM_DISABILITY_WEIGHT: [sequelae.moderate_wasting_with_edema,
                                                  sequelae.moderate_wasting_without_edema],
            data_keys.PEM.SAM_DISABILITY_WEIGHT: [sequelae.severe_wasting_with_edema,
                                                  sequelae.severe_wasting_without_edema],
        }

        prevalence_disability_weight = []
        state_prevalence = []
        for s in key_sequelae_map[key]:
            sequela_prevalence = interface.get_measure(s, 'prevalence', location)
            sequela_disability_weight = interface.get_measure(s, 'disability_weight', location)

            prevalence_disability_weight += [sequela_prevalence * sequela_disability_weight]
            state_prevalence += [sequela_prevalence]

        gbd_2019_disability_weight = (
            (sum(prevalence_disability_weight) / sum(state_prevalence))
            .fillna(0)
            .droplevel('location')
        )
        disability_weight = utilities.reshape_gbd_2019_data_as_gbd_2020_data(gbd_2019_disability_weight)
        return disability_weight
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_wasting_treatment_distribution(key: str, location: str) -> str:
    if key == data_keys.WASTING_TREATMENT.DISTRIBUTION:
        return data_values.WASTING.DISTRIBUTION
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_wasting_treatment_categories(key: str, location: str) -> str:
    if key == data_keys.WASTING_TREATMENT.CATEGORIES:
        return data_values.WASTING.CATEGORIES
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_wasting_treatment_exposure(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.WASTING_TREATMENT.EXPOSURE:
        treatment_coverage = get_random_variable_draws(pd.Index([f'draw_{i}' for i in range(0, 1000)]),
                                                       *data_values.WASTING.BASELINE_TX_COVERAGE)

        idx = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
        cat3 = pd.DataFrame({f'draw_{i}': 0.0 for i in range(0, 1000)}, index=idx)
        cat2 = pd.DataFrame({f'draw_{i}': 1.0 for i in range(0, 1000)}, index=idx) * treatment_coverage
        cat1 = 1 - cat2

        cat1['parameter'] = 'cat1'
        cat2['parameter'] = 'cat2'
        cat3['parameter'] = 'cat3'

        exposure = pd.concat([cat1, cat2, cat3]).set_index('parameter', append=True).sort_index()
        return exposure
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_wasting_treatment_rr(key: str, location: str) -> pd.DataFrame:
    # tmrel is defined as baseline treatment (cat_2)
    if key == data_keys.WASTING_TREATMENT.RELATIVE_RISK:
        idx_as_frame = (
            get_data(data_keys.POPULATION.DEMOGRAPHY, location).reset_index()
            .merge(pd.DataFrame({'parameter': [f'cat{i}' for i in range(1, 4)]}), how='cross')
        )
        idx = idx_as_frame.set_index(list(idx_as_frame.columns)).index

        raw_efficacy = {
            'baseline': {
                'sam': get_random_variable_draws(pd.Index([f'draw_{i}' for i in range(0, 1000)]),
                                                 *data_values.WASTING.BASELINE_SAM_TX_EFFICACY),
                'mam': get_random_variable_draws(pd.Index([f'draw_{i}' for i in range(0, 1000)]),
                                                 *data_values.WASTING.BASELINE_MAM_TX_EFFICACY)
            }, 'alternative': {
                'sam': data_values.WASTING.SAM_TX_ALTERNATIVE_EFFICACY,
                'mam': data_values.WASTING.MAM_TX_ALTERNATIVE_EFFICACY
            }
        }

        def get_treatment_efficacy(treatment_type: str) -> pd.DataFrame:
            efficacy = pd.DataFrame({f'draw_{i}': 1.0 for i in range(0, 1000)}, index=idx)
            efficacy[idx.get_level_values('parameter') == 'cat1'] *= 0.0
            efficacy[idx.get_level_values('parameter') == 'cat2'] *= raw_efficacy['baseline'][treatment_type]
            efficacy[idx.get_level_values('parameter') == 'cat3'] *= raw_efficacy['alternative'][treatment_type]
            return efficacy

        sam_tx_efficacy = get_treatment_efficacy('sam')
        mam_tx_efficacy = get_treatment_efficacy('mam')

        def get_tmrel_efficacy(efficacy: pd.DataFrame) -> pd.DataFrame:
            return efficacy[efficacy.index.get_level_values('parameter')
                            == data_keys.WASTING_TREATMENT.TMREL_CATEGORY].droplevel('parameter')

        sam_tx_efficacy_tmrel = get_tmrel_efficacy(sam_tx_efficacy)
        mam_tx_efficacy_tmrel = get_tmrel_efficacy(mam_tx_efficacy)

        mam_ux_duration = data_values.WASTING.MAM_UX_RECOVERY_TIME
        mam_tx_duration = pd.Series(index=idx)
        mam_tx_duration[idx.get_level_values('age_start') < 0.5] = data_values.WASTING.MAM_TX_RECOVERY_TIME_UNDER_6MO
        mam_tx_duration[0.5 <= idx.get_level_values('age_start')] = data_values.WASTING.MAM_TX_RECOVERY_TIME_OVER_6MO
        mam_tx_duration = (
            pd.DataFrame({f'draw_{i}': 1 for i in range(0, 1000)}, index=idx)
            .multiply(mam_tx_duration, axis='index')
        )

        # rr_t1 = t1 / t1_tmrel
        #       = (sam_tx_efficacy / sam_tx_duration) / (sam_tx_efficacy_tmrel / sam_tx_duration)
        #       = sam_tx_efficacy / sam_tx_efficacy_tmrel
        rr_sam_treated_remission = sam_tx_efficacy / sam_tx_efficacy_tmrel
        rr_sam_treated_remission['affected_entity'] = 'severe_acute_malnutrition_to_mild_child_wasting'

        # rr_r2 = r2 / r2_tmrel
        #       = (1 - sam_tx_efficacy) * (r2_ux) / (1 - sam_tx_efficacy_tmrel) * (r2_ux)
        #       = (1 - sam_tx_efficacy) / (1 - sam_tx_efficacy_tmrel)
        rr_sam_untreated_remission = (1 - sam_tx_efficacy) / (1 - sam_tx_efficacy_tmrel)
        rr_sam_untreated_remission['affected_entity'] = 'severe_acute_malnutrition_to_moderate_acute_malnutrition'

        # rr_r3 = r3 / r3_tmrel
        #       = (mam_tx_efficacy / mam_tx_duration) + (1 - mam_tx_efficacy / mam_ux_duration)
        #           / (mam_tx_efficacy_tmrel / mam_tx_duration) + (1 - mam_tx_efficacy_tmrel / mam_ux_duration)
        #       = (mam_tx_efficacy * mam_ux_duration + (1 - mam_tx_efficacy) * mam_tx_duration)
        #           / (mam_tx_efficacy_tmrel * mam_ux_duration + (1 - mam_tx_efficacy_tmrel) * mam_tx_duration)
        rr_mam_remission = ((mam_tx_efficacy * mam_ux_duration + (1 - mam_tx_efficacy) * mam_tx_duration)
                            / (mam_tx_efficacy_tmrel * mam_ux_duration + (1 - mam_tx_efficacy_tmrel) * mam_tx_duration))
        rr_mam_remission['affected_entity'] = 'moderate_acute_malnutrition_to_mild_child_wasting'

        rr = pd.concat(
            [rr_sam_treated_remission, rr_sam_untreated_remission, rr_mam_remission]
        )
        rr['affected_measure'] = 'transition_rate'
        rr = rr.set_index(['affected_entity', 'affected_measure'], append=True)
        rr.index = rr.index.reorder_levels([col for col in rr.index.names if col != 'parameter'] + ['parameter'])
        rr.sort_index()
        return rr
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_low_maternal_bmi_distribution(key: str, location: str) -> str:
    if key in [data_keys.X_FACTOR.DISTRIBUTION]:
        return data_values.MATERNAL_BMI.DISTRIBUTION
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_low_maternal_bmi_categories(key: str, location: str) -> str:
    if key in [data_keys.X_FACTOR.CATEGORIES]:
        return data_values.MATERNAL_BMI.CATEGORIES
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_low_maternal_bmi_exposure(key: str, location: str) -> pd.DataFrame:
    if key in [data_keys.X_FACTOR.EXPOSURE]:
        exposure = get_random_variable_draws(pd.Index([f'draw_{i}' for i in range(0, 1000)]),
                                             *data_values.MATERNAL_BMI.EXPOSURE)

        idx = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
        cat1 = pd.DataFrame({f'draw_{i}': 1.0 for i in range(0, 1000)}, index=idx) * exposure
        cat2 = 1 - cat1

        cat1['parameter'] = 'cat1'
        cat2['parameter'] = 'cat2'

        exposure = pd.concat([cat1, cat2]).set_index('parameter', append=True).sort_index()
        return exposure
    else:
        raise ValueError(f'Unrecognized key {key}')
