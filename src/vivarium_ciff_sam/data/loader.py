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
import pickle
from typing import Dict, Tuple, Type, Union

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RectBivariateSpline
from scipy import stats

from gbd_mapping import sequelae, Cause
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import constants as gbd_constants
from vivarium_inputs import interface

from vivarium_ciff_sam.components import LBWSGSubRisk, LowBirthWeight, ShortGestation
from vivarium_ciff_sam.constants import data_keys, data_values, metadata, paths
from vivarium_ciff_sam.constants.metadata import ARTIFACT_COLUMNS
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
        data_keys.DIARRHEA.REMISSION_RATE: load_remission_rate_from_duration,
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
        data_keys.LRI.REMISSION_RATE: load_remission_rate_from_duration,
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

        data_keys.SAM_TREATMENT.EXPOSURE: load_wasting_treatment_exposure,
        data_keys.SAM_TREATMENT.DISTRIBUTION: load_wasting_treatment_distribution,
        data_keys.SAM_TREATMENT.CATEGORIES: load_wasting_treatment_categories,
        data_keys.SAM_TREATMENT.RELATIVE_RISK: load_sam_treatment_rr,
        data_keys.SAM_TREATMENT.PAF: load_paf,

        data_keys.MAM_TREATMENT.EXPOSURE: load_wasting_treatment_exposure,
        data_keys.MAM_TREATMENT.DISTRIBUTION: load_wasting_treatment_distribution,
        data_keys.MAM_TREATMENT.CATEGORIES: load_wasting_treatment_categories,
        data_keys.MAM_TREATMENT.RELATIVE_RISK: load_mam_treatment_rr,
        data_keys.MAM_TREATMENT.PAF: load_paf,

        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.EXPOSURE: load_lbwsg_exposure,
        data_keys.LBWSG.RELATIVE_RISK: load_lbwsg_rr,
        data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR: load_lbwsg_interpolated_rr,
        data_keys.LBWSG.PAF: load_lbwsg_paf,

        data_keys.NON_EXCLUSIVE_BREASTFEEDING.DISTRIBUTION: load_metadata,
        data_keys.NON_EXCLUSIVE_BREASTFEEDING.CATEGORIES: load_metadata,
        data_keys.NON_EXCLUSIVE_BREASTFEEDING.EXPOSURE: load_gbd_2020_exposure,
        data_keys.NON_EXCLUSIVE_BREASTFEEDING.RELATIVE_RISK: load_gbd_2020_rr,
        data_keys.NON_EXCLUSIVE_BREASTFEEDING.PAF: load_paf,

        data_keys.DISCONTINUED_BREASTFEEDING.DISTRIBUTION: load_metadata,
        data_keys.DISCONTINUED_BREASTFEEDING.CATEGORIES: load_metadata,
        data_keys.DISCONTINUED_BREASTFEEDING.EXPOSURE: load_gbd_2020_exposure,
        data_keys.DISCONTINUED_BREASTFEEDING.RELATIVE_RISK: load_gbd_2020_rr,
        data_keys.DISCONTINUED_BREASTFEEDING.PAF: load_paf,

        data_keys.AFFECTED_UNMODELED_CAUSES.URI_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.OTITIS_MEDIA_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.MENINGITIS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.ENCEPHALITIS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_PRETERM_BIRTH_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_ENCEPHALOPATHY_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_SEPSIS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_JAUNDICE_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.OTHER_NEONATAL_DISORDERS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.SIDS_CSMR: load_sids_csmr,

        data_keys.MATERNAL_MALNUTRITION.DISTRIBUTION: load_maternal_malnutrition_distribution,
        data_keys.MATERNAL_MALNUTRITION.CATEGORIES: load_maternal_malnutrition_categories,
        data_keys.MATERNAL_MALNUTRITION.EXPOSURE: load_dichotomous_risk_exposure,
        data_keys.MATERNAL_MALNUTRITION.EXCESS_SHIFT: load_risk_excess_shift,
        data_keys.MATERNAL_MALNUTRITION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,

        data_keys.IFA_SUPPLEMENTATION.DISTRIBUTION: load_intervention_distribution,
        data_keys.IFA_SUPPLEMENTATION.CATEGORIES: load_intervention_categories,
        data_keys.IFA_SUPPLEMENTATION.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT: load_treatment_excess_shift,
        data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,

        data_keys.MMN_SUPPLEMENTATION.DISTRIBUTION: load_intervention_distribution,
        data_keys.MMN_SUPPLEMENTATION.CATEGORIES: load_intervention_categories,
        data_keys.MMN_SUPPLEMENTATION.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.MMN_SUPPLEMENTATION.EXCESS_SHIFT: load_treatment_excess_shift,
        data_keys.MMN_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,

        data_keys.BEP_SUPPLEMENTATION.DISTRIBUTION: load_intervention_distribution,
        data_keys.BEP_SUPPLEMENTATION.CATEGORIES: load_intervention_categories,
        data_keys.BEP_SUPPLEMENTATION.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.BEP_SUPPLEMENTATION.EXCESS_SHIFT: load_treatment_excess_shift,
        data_keys.BEP_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,

        data_keys.INSECTICIDE_TX_NETS.DISTRIBUTION: load_intervention_distribution,
        data_keys.INSECTICIDE_TX_NETS.CATEGORIES: load_intervention_categories,
        data_keys.INSECTICIDE_TX_NETS.EXPOSURE: load_insecticide_treated_nets_exposure,
        data_keys.INSECTICIDE_TX_NETS.EXCESS_SHIFT: load_treatment_excess_shift,
        data_keys.INSECTICIDE_TX_NETS.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,

        data_keys.PREVENTATIVE_ZINC.DISTRIBUTION: load_intervention_distribution,
        data_keys.PREVENTATIVE_ZINC.CATEGORIES: load_intervention_categories,
        data_keys.PREVENTATIVE_ZINC.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.PREVENTATIVE_ZINC.RELATIVE_RISK: load_treatment_rr,
        data_keys.PREVENTATIVE_ZINC.PAF: load_paf,

        data_keys.THERAPEUTIC_ZINC.DISTRIBUTION: load_intervention_distribution,
        data_keys.THERAPEUTIC_ZINC.CATEGORIES: load_intervention_categories,
        data_keys.THERAPEUTIC_ZINC.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.THERAPEUTIC_ZINC.RELATIVE_RISK: load_treatment_rr,
        data_keys.THERAPEUTIC_ZINC.PAF: load_paf,
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
        utilities.get_gbd_age_bins(metadata.AGE_GROUP.GBD_2020)
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


def load_remission_rate_from_duration(key: str, location: str) -> pd.DataFrame:
    try:
        distribution = {
            data_keys.DIARRHEA.REMISSION_RATE: data_values.DIARRHEA_DURATION,
            data_keys.LRI.REMISSION_RATE: data_values.LRI_DURATION,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    duration = (
            get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *distribution)
            / data_values.YEAR_DURATION
    )
    remission_rate = pd.DataFrame([1 / duration], index=index)
    return remission_rate


def load_lri_prevalence(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.LRI.PREVALENCE:
        incidence_rate = get_data(data_keys.LRI.INCIDENCE_RATE, location)
        duration = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *data_values.LRI_DURATION)
        early_neonatal_prevalence = (
                incidence_rate.query(f'age_start == 0.0')
                * data_values.EARLY_NEONATAL_CAUSE_DURATION / data_values.YEAR_DURATION
        )
        all_other_prevalence = (
                incidence_rate.query(f'age_start > 0.0')
                * duration / data_values.YEAR_DURATION
        )
        prevalence = pd.concat([early_neonatal_prevalence, all_other_prevalence]).sort_index()
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
    entity_key = EntityKey(key)
    entity = utilities.get_gbd_2020_entity(entity_key)

    data = utilities.get_data(entity_key, entity, location, gbd_constants.SOURCES.EXPOSURE, 'rei_id',
                              metadata.AGE_GROUP.GBD_2020, metadata.GBD_2020_ROUND_ID)
    data = utilities.process_exposure(data, entity_key, entity, location, metadata.GBD_2020_ROUND_ID,
                                      metadata.AGE_GROUP.GBD_2020)

    if entity_key == data_keys.STUNTING.EXPOSURE:
        # Remove neonatal exposure
        neonatal_age_ends = data.index.get_level_values('age_end').unique()[:2]
        data.loc[data.index.get_level_values('age_end').isin(neonatal_age_ends)] = 0.0
        data.loc[data.index.get_level_values('age_end').isin(neonatal_age_ends)
                 & (data.index.get_level_values('parameter') == data_keys.STUNTING.CAT4)] = 1.0
    return data


def load_gbd_2020_rr(key: str, location: str) -> pd.DataFrame:
    entity_key = EntityKey(key)
    entity = utilities.get_gbd_2020_entity(entity_key)

    data = utilities.get_data(
        entity_key,
        entity,
        location,
        gbd_constants.SOURCES.RR,
        'rei_id',
        metadata.AGE_GROUP.GBD_2020,
        metadata.GBD_2020_ROUND_ID
    )
    data = utilities.process_relative_risk(data, entity_key, entity, location, metadata.GBD_2020_ROUND_ID,
                                           metadata.AGE_GROUP.GBD_2020)

    if key == data_keys.STUNTING.RELATIVE_RISK:
        # Remove neonatal relative risks
        neonatal_age_ends = data.index.get_level_values('age_end').unique()[:2]
        data.loc[data.index.get_level_values('age_end').isin(neonatal_age_ends)] = 1.0
    elif key == data_keys.WASTING.RELATIVE_RISK:
        # Remove relative risks for simulants under 6 months
        data.loc[data.index.get_level_values('age_end') <= data_values.WASTING.START_AGE] = 1.0

        # Set risk to affect diarrheal emr
        diarrhea_rr = data.query(f"affected_entity == '{data_keys.DIARRHEA.name}'")
        data = pd.concat([
            diarrhea_rr.rename(
                index={'incidence_rate': 'excess_mortality_rate'}, level='affected_measure'
            ), data.drop(diarrhea_rr.index)
        ]).sort_index()
    elif key == data_keys.DISCONTINUED_BREASTFEEDING.RELATIVE_RISK:
        # Remove RR outside of [6 months, 2 years)
        discontinued_tmrel_index = data.query(
            f'age_start < {data_values.DISCONTINUED_BREASTFEEDING_START_AGE}'
            f' or age_end > {data_values.DISCONTINUED_BREASTFEEDING_END_AGE}'
        ).index
        discontinued_tmrel_rr = pd.DataFrame(
            1.0, columns=metadata.ARTIFACT_COLUMNS, index=discontinued_tmrel_index
        )
        data.update(discontinued_tmrel_rr)
    elif key == data_keys.NON_EXCLUSIVE_BREASTFEEDING.RELATIVE_RISK:
        # Remove month [6, months, 1 year) exposure
        non_exclusive_tmrel_index = data.query(
            f'age_start == {data_values.NON_EXCLUSIVE_BREASTFEEDING_END_AGE}'
        ).index
        non_exclusive_tmrel_rr = pd.DataFrame(
            1.0, columns=metadata.ARTIFACT_COLUMNS, index=non_exclusive_tmrel_index
        )
        data.update(non_exclusive_tmrel_rr)
    return data


def load_treatment_rr(key: str, location: str) -> pd.DataFrame:
    try:
        distribution = {
            data_keys.PREVENTATIVE_ZINC.RELATIVE_RISK: data_values.PREVENTATIVE_ZINC.PREVENTATIVE_TX_EFFICACY,
            data_keys.THERAPEUTIC_ZINC.RELATIVE_RISK: None,
        }[key]
        affected_entity = {
            data_keys.PREVENTATIVE_ZINC.RELATIVE_RISK: data_keys.PREVENTATIVE_ZINC.AFFECTED_ENTITY,
            data_keys.THERAPEUTIC_ZINC.RELATIVE_RISK: data_keys.THERAPEUTIC_ZINC.AFFECTED_ENTITY,
        }[key]
        affected_measure = {
            data_keys.PREVENTATIVE_ZINC.RELATIVE_RISK: data_keys.PREVENTATIVE_ZINC.AFFECTED_MEASURE,
            data_keys.THERAPEUTIC_ZINC.RELATIVE_RISK: data_keys.THERAPEUTIC_ZINC.AFFECTED_MEASURE,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    if key == data_keys.THERAPEUTIC_ZINC.RELATIVE_RISK:
        cat2_rr = calculate_therapeutic_zinc_distribution(key)
    else:
        cat2_rr = get_random_variable_draws(
            metadata.ARTIFACT_COLUMNS, *distribution
        )

    exposed = pd.DataFrame([cat2_rr], index=index)
    exposed['parameter'] = 'cat2'
    unexposed = pd.DataFrame([pd.Series(1.0, index=metadata.ARTIFACT_COLUMNS)], index=index)
    unexposed['parameter'] = 'cat1'

    rr = pd.concat([exposed, unexposed])
    rr['affected_entity'] = affected_entity
    rr['affected_measure'] = affected_measure

    rr = (
        rr.set_index(['affected_entity', 'affected_measure', 'parameter'], append=True).sort_index()
    )

    return rr


def calculate_therapeutic_zinc_distribution(key: str) -> pd.Series:
    if key != data_keys.THERAPEUTIC_ZINC.RELATIVE_RISK:
        raise ValueError(f'Unrecognized key {key}')

    diarrhea_duration_shift_years = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS, *data_values.THERAPEUTIC_ZINC.DIARRHEA_DURATION_SHIFT_HOURS
    ) / (data_values.DAY_DURATION * data_values.YEAR_DURATION)

    diarrhea_duration_years = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *data_values.DIARRHEA_DURATION) / (
            data_values.DAY_DURATION * data_values.YEAR_DURATION
    )

    baseline_coverage = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS, *data_values.THERAPEUTIC_ZINC.BASELINE_THERAPEUTIC_COVERAGE
    )
    duration_uncovered = diarrhea_duration_years - (
            diarrhea_duration_shift_years * baseline_coverage
    )
    duration_covered = duration_uncovered + diarrhea_duration_shift_years

    remission_rate_uncovered = 1 / duration_uncovered
    remission_rate_covered = 1 / duration_covered

    rr = remission_rate_covered / remission_rate_uncovered
    return rr


def load_paf(key: str, location: str) -> pd.DataFrame:
    try:
        risk = {
            data_keys.WASTING.PAF: data_keys.WASTING,
            data_keys.STUNTING.PAF: data_keys.STUNTING,
            data_keys.SAM_TREATMENT.PAF: data_keys.SAM_TREATMENT,
            data_keys.MAM_TREATMENT.PAF: data_keys.MAM_TREATMENT,
            data_keys.DISCONTINUED_BREASTFEEDING.PAF: data_keys.DISCONTINUED_BREASTFEEDING,
            data_keys.NON_EXCLUSIVE_BREASTFEEDING.PAF: data_keys.NON_EXCLUSIVE_BREASTFEEDING,
            data_keys.PREVENTATIVE_ZINC.PAF: data_keys.PREVENTATIVE_ZINC,
            data_keys.THERAPEUTIC_ZINC.PAF: data_keys.THERAPEUTIC_ZINC,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')

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


def load_pem_disability_weight(key: str, location: str) -> pd.DataFrame:
    try:
        pem_sequelae = {
            data_keys.PEM.MAM_DISABILITY_WEIGHT: [sequelae.moderate_wasting_with_edema,
                                                  sequelae.moderate_wasting_without_edema],
            data_keys.PEM.SAM_DISABILITY_WEIGHT: [sequelae.severe_wasting_with_edema,
                                                  sequelae.severe_wasting_without_edema],
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')

    prevalence_disability_weight = []
    state_prevalence = []
    for s in pem_sequelae:
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


# noinspection PyUnusedLocal
def load_wasting_treatment_distribution(key: str, location: str) -> str:
    if key in [data_keys.SAM_TREATMENT.DISTRIBUTION, data_keys.MAM_TREATMENT.DISTRIBUTION]:
        return data_values.WASTING.DISTRIBUTION
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_wasting_treatment_categories(key: str, location: str) -> str:
    if key in [data_keys.SAM_TREATMENT.CATEGORIES, data_keys.MAM_TREATMENT.CATEGORIES]:
        return data_values.WASTING.CATEGORIES
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_wasting_treatment_exposure(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.SAM_TREATMENT.EXPOSURE:
        coverage_distribution = data_values.WASTING.BASELINE_SAM_TX_COVERAGE
    elif key == data_keys.MAM_TREATMENT.EXPOSURE:
        coverage_distribution = data_values.WASTING.BASELINE_MAM_TX_COVERAGE
    else:
        raise ValueError(f'Unrecognized key {key}')

    treatment_coverage = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS, *coverage_distribution
    )

    idx = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    cat3 = pd.DataFrame({f'draw_{i}': 0.0 for i in range(0, 1000)}, index=idx)
    cat2 = pd.DataFrame({f'draw_{i}': 1.0 for i in range(0, 1000)}, index=idx) * treatment_coverage
    cat1 = 1 - cat2

    cat1['parameter'] = 'cat1'
    cat2['parameter'] = 'cat2'
    cat3['parameter'] = 'cat3'

    exposure = pd.concat([cat1, cat2, cat3]).set_index('parameter', append=True).sort_index()
    return exposure


def load_sam_treatment_rr(key: str, location: str) -> pd.DataFrame:
    # tmrel is defined as baseline treatment (cat_2)
    if key != data_keys.SAM_TREATMENT.RELATIVE_RISK:
        raise ValueError(f'Unrecognized key {key}')

    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location).reset_index()
    sam_tx_efficacy, sam_tx_efficacy_tmrel = utilities.get_treatment_efficacy(demography, data_keys.WASTING.CAT1)

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

    rr = pd.concat(
        [rr_sam_treated_remission, rr_sam_untreated_remission]
    )
    rr['affected_measure'] = 'transition_rate'
    rr = rr.set_index(['affected_entity', 'affected_measure'], append=True)
    rr.index = rr.index.reorder_levels([col for col in rr.index.names if col != 'parameter'] + ['parameter'])
    rr.sort_index()
    return rr


def load_mam_treatment_rr(key: str, location: str) -> pd.DataFrame:
    # tmrel is defined as baseline treatment (cat_2)
    if key != data_keys.MAM_TREATMENT.RELATIVE_RISK:
        raise ValueError(f'Unrecognized key {key}')

    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location).reset_index()
    mam_tx_efficacy, mam_tx_efficacy_tmrel = utilities.get_treatment_efficacy(demography, data_keys.WASTING.CAT2)
    index = mam_tx_efficacy.index

    mam_ux_duration = data_values.WASTING.MAM_UX_RECOVERY_TIME
    mam_tx_duration = pd.Series(index=index)
    mam_tx_duration[index.get_level_values('age_start') < 0.5] = data_values.WASTING.MAM_TX_RECOVERY_TIME_UNDER_6MO
    mam_tx_duration[0.5 <= index.get_level_values('age_start')] = data_values.WASTING.MAM_TX_RECOVERY_TIME_OVER_6MO
    mam_tx_duration = (
        pd.DataFrame({f'draw_{i}': 1 for i in range(0, 1000)}, index=index)
            .multiply(mam_tx_duration, axis='index')
    )

    # rr_r3 = r3 / r3_tmrel
    #       = (mam_tx_efficacy / mam_tx_duration) + (1 - mam_tx_efficacy / mam_ux_duration)
    #           / (mam_tx_efficacy_tmrel / mam_tx_duration) + (1 - mam_tx_efficacy_tmrel / mam_ux_duration)
    #       = (mam_tx_efficacy * mam_ux_duration + (1 - mam_tx_efficacy) * mam_tx_duration)
    #           / (mam_tx_efficacy_tmrel * mam_ux_duration + (1 - mam_tx_efficacy_tmrel) * mam_tx_duration)
    rr = ((mam_tx_efficacy * mam_ux_duration + (1 - mam_tx_efficacy) * mam_tx_duration)
          / (mam_tx_efficacy_tmrel * mam_ux_duration + (1 - mam_tx_efficacy_tmrel) * mam_tx_duration))

    rr['affected_entity'] = 'moderate_acute_malnutrition_to_mild_child_wasting'
    rr['affected_measure'] = 'transition_rate'
    rr = rr.set_index(['affected_entity', 'affected_measure'], append=True)
    rr.index = rr.index.reorder_levels([col for col in rr.index.names if col != 'parameter'] + ['parameter'])
    rr.sort_index()
    return rr


def load_lbwsg_exposure(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.EXPOSURE:
        raise ValueError(f'Unrecognized key {key}')

    key = EntityKey(key)
    entity = utilities.get_entity(key)
    data = utilities.get_data(key, entity, location, gbd_constants.SOURCES.EXPOSURE, 'rei_id',
                              metadata.AGE_GROUP.GBD_2019_LBWSG_EXPOSURE, metadata.GBD_2019_ROUND_ID, 'step4')
    data = data[data['year_id'] == 2019].drop(columns='year_id')
    data = utilities.process_exposure(data, key, entity, location, metadata.GBD_2019_ROUND_ID,
                                      metadata.AGE_GROUP.GBD_2019_LBWSG_EXPOSURE | metadata.AGE_GROUP.GBD_2020)
    data = data[data.index.get_level_values('year_start') == 2019]
    return data


def load_lbwsg_rr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.RELATIVE_RISK:
        raise ValueError(f'Unrecognized key {key}')

    key = EntityKey(key)
    entity = utilities.get_entity(key)
    data = utilities.get_data(key, entity, location, gbd_constants.SOURCES.RR, 'rei_id',
                              metadata.AGE_GROUP.GBD_2019_LBWSG_RELATIVE_RISK, metadata.GBD_2019_ROUND_ID, 'step4')
    data = data[data['year_id'] == 2019].drop(columns='year_id')
    data = utilities.process_relative_risk(data, key, entity, location, metadata.GBD_2019_ROUND_ID,
                                           metadata.AGE_GROUP.GBD_2020, whitelist_sids=True)
    data = (
        data.query('year_start == 2019')
            .droplevel(['affected_entity', 'affected_measure'])
    )
    data = data[~data.index.duplicated()]
    return data


def load_lbwsg_interpolated_rr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR:
        raise ValueError(f'Unrecognized key {key}')

    rr = get_data(data_keys.LBWSG.RELATIVE_RISK, location).reset_index()
    rr['parameter'] = pd.Categorical(rr['parameter'], [f'cat{i}' for i in range(1000)])
    rr = (
        rr.sort_values('parameter')
            .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ['parameter'])
            .stack()
            .unstack('parameter')
            .apply(np.log)
    )

    # get category midpoints
    def get_category_midpoints(lbwsg_type: Type[LBWSGSubRisk]) -> pd.Series:
        categories = get_data(f'risk_factor.{data_keys.LBWSG.name}.categories', location)
        return lbwsg_type.get_intervals_from_categories(categories).apply(lambda x: x.mid)

    gestational_age_midpoints = get_category_midpoints(ShortGestation)
    birth_weight_midpoints = get_category_midpoints(LowBirthWeight)

    # build grid of gestational age and birth weight
    def get_grid(midpoints: pd.Series, endpoints: Tuple[float, float]) -> np.array:
        grid = np.append(np.unique(midpoints), endpoints)
        grid.sort()
        return grid

    gestational_age_grid = get_grid(gestational_age_midpoints, (0.0, 42.0))
    birth_weight_grid = get_grid(birth_weight_midpoints, (0.0, 4500.0))

    def make_interpolator(log_rr_for_age_sex_draw: pd.Series) -> RectBivariateSpline:
        # Use scipy.interpolate.griddata to extrapolate to grid using nearest neighbor interpolation
        log_rr_grid_nearest = griddata(
            (gestational_age_midpoints, birth_weight_midpoints),
            log_rr_for_age_sex_draw,
            (gestational_age_grid[:, None], birth_weight_grid[None, :]),
            method='nearest',
            rescale=True
        )
        # return a RectBivariateSpline object from the extrapolated values on grid
        return RectBivariateSpline(gestational_age_grid, birth_weight_grid, log_rr_grid_nearest, kx=1, ky=1)

    log_rr_interpolator = (
        rr.apply(make_interpolator, axis='columns')
            .apply(lambda x: pickle.dumps(x).hex())
            .unstack()
    )
    return log_rr_interpolator


def load_lbwsg_paf(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.PAF:
        raise ValueError(f'Unrecognized key {key}')

    paf_files = paths.TEMPORARY_PAF_DIR.glob('*.hdf')
    paf_data = (
        pd.concat([pd.read_hdf(paf_file) for paf_file in paf_files])
            .sort_values(metadata.ARTIFACT_INDEX_COLUMNS + ['draw'])
    )

    paf_data['draw'] = paf_data['draw'].apply(lambda draw: f'draw_{draw}')

    paf_data = (
        paf_data.set_index(metadata.ARTIFACT_INDEX_COLUMNS + ['draw'])
            .unstack()
    )

    paf_data.columns = paf_data.columns.droplevel(0)
    paf_data.columns.name = None

    full_index = (
        get_data(data_keys.LBWSG.RELATIVE_RISK, location).index
            .droplevel('parameter')
            .drop_duplicates()
    )

    paf_data = (
        paf_data.reindex(full_index)
            .fillna(0.0)
    )
    return paf_data


def load_sids_csmr(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.AFFECTED_UNMODELED_CAUSES.SIDS_CSMR:
        key = EntityKey(key)
        entity: Cause = utilities.get_entity(key)

        # get around the validation rejecting yll only causes
        entity.restrictions.yll_only = False
        entity.restrictions.yld_age_group_id_start = min(metadata.AGE_GROUP.GBD_2019_SIDS)
        entity.restrictions.yld_age_group_id_end = max(metadata.AGE_GROUP.GBD_2019_SIDS)

        data = interface.get_measure(entity, key.measure, location).droplevel('location')
        return data
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_maternal_malnutrition_distribution(key: str, location: str) -> str:
    if key != data_keys.MATERNAL_MALNUTRITION.DISTRIBUTION:
        raise ValueError(f'Unrecognized key {key}')

    return 'dichotomous'


# noinspection PyUnusedLocal
def load_maternal_malnutrition_categories(key: str, location: str) -> Dict[str, str]:
    if key != data_keys.MATERNAL_MALNUTRITION.CATEGORIES:
        raise ValueError(f'Unrecognized key {key}')

    return {
        'cat1': 'BMI < 18.5',
        'cat2': 'BMI >= 18.5',
    }


def load_dichotomous_risk_exposure(key: str, location: str, **kwargs) -> pd.DataFrame:
    try:
        distribution_data = {
            data_keys.MATERNAL_MALNUTRITION.EXPOSURE: data_values.MATERNAL_MALNUTRITION.EXPOSURE,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
    return load_dichotomous_exposure(location, distribution_data, is_risk=True, **kwargs)


def load_dichotomous_treatment_exposure(key: str, location: str, **kwargs) -> pd.DataFrame:
    try:
        distribution_data = {
            data_keys.INSECTICIDE_TX_NETS.EXPOSURE: data_values.INSECTICIDE_TX_NETS.EXPOSURE,
            data_keys.IFA_SUPPLEMENTATION.EXPOSURE: data_values.MATERNAL_SUPPLEMENTATION.BASELINE_IFA_COVERAGE,
            data_keys.MMN_SUPPLEMENTATION.EXPOSURE: data_values.MATERNAL_SUPPLEMENTATION.BASELINE_MMN_COVERAGE,
            data_keys.BEP_SUPPLEMENTATION.EXPOSURE: data_values.MATERNAL_SUPPLEMENTATION.BASELINE_BEP_COVERAGE,
            data_keys.PREVENTATIVE_ZINC.EXPOSURE: data_values.PREVENTATIVE_ZINC.BASELINE_PREVENTATIVE_COVERAGE,
            data_keys.THERAPEUTIC_ZINC.EXPOSURE: data_values.THERAPEUTIC_ZINC.BASELINE_THERAPEUTIC_COVERAGE,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
    return load_dichotomous_exposure(location, distribution_data, is_risk=False, **kwargs)


def load_insecticide_treated_nets_exposure(key: str, location: str) -> pd.DataFrame:
    exposure = load_dichotomous_treatment_exposure(
        key,
        location,
        coverage=data_values.INSECTICIDE_TX_NETS.PROP_MALARIOUS
    )

    return exposure


def load_dichotomous_exposure(
        location: str, distribution_data: Union[float, Tuple], is_risk: bool, coverage: float = 1.0,
) -> pd.DataFrame:
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    if type(distribution_data) == float:
        base_exposure = pd.Series(distribution_data, index=metadata.ARTIFACT_COLUMNS)
    else:
        base_exposure = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *distribution_data)

    exposed = pd.DataFrame([base_exposure * coverage], index=index)
    unexposed = 1 - exposed

    exposed['parameter'] = 'cat1' if is_risk else 'cat2'
    unexposed['parameter'] = 'cat2' if is_risk else 'cat1'

    exposure = pd.concat([exposed, unexposed]).set_index('parameter', append=True).sort_index()
    return exposure


def load_risk_excess_shift(key: str, location: str) -> pd.DataFrame:
    try:
        distribution_data = {
            data_keys.MATERNAL_MALNUTRITION.EXCESS_SHIFT:
                data_values.MATERNAL_MALNUTRITION.EXPOSED_BIRTH_WEIGHT_SHIFT,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
    return load_dichotomous_excess_shift(location, distribution_data, is_risk=True)


def load_treatment_excess_shift(key: str, location: str) -> pd.DataFrame:
    try:
        distribution_data = {
            data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT:
                data_values.MATERNAL_SUPPLEMENTATION.IFA_BIRTH_WEIGHT_SHIFT,
            data_keys.MMN_SUPPLEMENTATION.EXCESS_SHIFT:
                data_values.MATERNAL_SUPPLEMENTATION.MMN_BIRTH_WEIGHT_SHIFT,
            data_keys.BEP_SUPPLEMENTATION.EXCESS_SHIFT:
                data_values.MATERNAL_SUPPLEMENTATION.BEP_BIRTH_WEIGHT_SHIFT,
            data_keys.INSECTICIDE_TX_NETS.EXCESS_SHIFT:
                data_values.INSECTICIDE_TX_NETS.BIRTH_WEIGHT_SHIFT,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
    return load_dichotomous_excess_shift(location, distribution_data, is_risk=False)


def load_dichotomous_excess_shift(
        location: str, distribution_data: Tuple, is_risk: bool
) -> pd.DataFrame:
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    shift = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *distribution_data)

    exposed = pd.DataFrame([shift], index=index)
    exposed['parameter'] = 'cat1' if is_risk else 'cat2'
    unexposed = pd.DataFrame([pd.Series(0.0, index=metadata.ARTIFACT_COLUMNS)], index=index)
    unexposed['parameter'] = 'cat2' if is_risk else 'cat1'

    excess_shift = pd.concat([exposed, unexposed])
    excess_shift['affected_entity'] = data_keys.LBWSG.BIRTH_WEIGHT_EXPOSURE.name
    excess_shift['affected_measure'] = data_keys.LBWSG.BIRTH_WEIGHT_EXPOSURE.measure

    excess_shift = (
        excess_shift
            .set_index(['affected_entity', 'affected_measure', 'parameter'], append=True)
            .sort_index()
    )
    return excess_shift


def load_risk_specific_shift(key: str, location: str) -> pd.DataFrame:
    try:
        key_group: data_keys.__AdditiveRisk = {
            data_keys.MATERNAL_MALNUTRITION.RISK_SPECIFIC_SHIFT: data_keys.MATERNAL_MALNUTRITION,
            data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.IFA_SUPPLEMENTATION,
            data_keys.MMN_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.MMN_SUPPLEMENTATION,
            data_keys.BEP_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.BEP_SUPPLEMENTATION,
            data_keys.INSECTICIDE_TX_NETS.RISK_SPECIFIC_SHIFT: data_keys.INSECTICIDE_TX_NETS,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')

    # p_exposed * exposed_shift
    exposure = (
        get_data(key_group.EXPOSURE, location)
    )
    excess_shift = (
        get_data(key_group.EXCESS_SHIFT, location)
    )

    risk_specific_shift = (
        (exposure * excess_shift)
            .groupby(metadata.ARTIFACT_INDEX_COLUMNS + ['affected_entity', 'affected_measure'])
            .sum()
    )
    return risk_specific_shift


# noinspection PyUnusedLocal
def load_intervention_distribution(key: str, location: str) -> str:
    try:
        return {
            data_keys.INSECTICIDE_TX_NETS.DISTRIBUTION: data_values.INSECTICIDE_TX_NETS.DISTRIBUTION,
            data_keys.IFA_SUPPLEMENTATION.DISTRIBUTION: data_values.MATERNAL_SUPPLEMENTATION.DISTRIBUTION,
            data_keys.MMN_SUPPLEMENTATION.DISTRIBUTION: data_values.MATERNAL_SUPPLEMENTATION.DISTRIBUTION,
            data_keys.BEP_SUPPLEMENTATION.DISTRIBUTION: data_values.MATERNAL_SUPPLEMENTATION.DISTRIBUTION,
            data_keys.PREVENTATIVE_ZINC.DISTRIBUTION: data_values.PREVENTATIVE_ZINC.DISTRIBUTION,
            data_keys.THERAPEUTIC_ZINC.DISTRIBUTION: data_values.THERAPEUTIC_ZINC.DISTRIBUTION,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_intervention_categories(key: str, location: str) -> str:
    try:
        return {
            data_keys.INSECTICIDE_TX_NETS.CATEGORIES: data_values.INSECTICIDE_TX_NETS.CATEGORIES,
            data_keys.IFA_SUPPLEMENTATION.CATEGORIES: data_values.MATERNAL_SUPPLEMENTATION.CATEGORIES,
            data_keys.MMN_SUPPLEMENTATION.CATEGORIES: data_values.MATERNAL_SUPPLEMENTATION.CATEGORIES,
            data_keys.BEP_SUPPLEMENTATION.CATEGORIES: data_values.MATERNAL_SUPPLEMENTATION.CATEGORIES,
            data_keys.PREVENTATIVE_ZINC.CATEGORIES: data_values.PREVENTATIVE_ZINC.CATEGORIES,
            data_keys.THERAPEUTIC_ZINC.CATEGORIES: data_values.THERAPEUTIC_ZINC.CATEGORIES,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
