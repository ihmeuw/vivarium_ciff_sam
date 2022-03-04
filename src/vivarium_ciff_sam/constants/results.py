import itertools

from vivarium_ciff_sam.constants import models

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

# Columns from parallel runs
INPUT_DRAW_COLUMN = 'input_draw'
RANDOM_SEED_COLUMN = 'random_seed'

OUTPUT_INPUT_DRAW_COLUMN = 'input_data.input_draw_number'
OUTPUT_RANDOM_SEED_COLUMN = 'randomness.random_seed'
OUTPUT_SCENARIO_COLUMN = 'intervention.scenario'
X_FACTOR_EFFECT_COLUMN = 'effect_of_x_factor_on_mild_child_wasting.incidence_rate.relative_risk'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
DEATH_COLUMN_TEMPLATE = 'death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
YLLS_COLUMN_TEMPLATE = 'ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
YLDS_COLUMN_TEMPLATE = 'ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
DISEASE_STATE_PERSON_TIME_COLUMN_TEMPLATE = '{DISEASE_STATE_EXCL_DIARRHEA}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
DISEASE_TRANSITION_COUNT_COLUMN_TEMPLATE = '{DISEASE_TRANSITION_EXCL_DIARRHEA}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
WASTING_STATE_PERSON_TIME_COLUMN_TEMPLATE = '{WASTING_STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_sam_treatment_{SAM_TREATMENT_STATE}_mam_treatment_{MAM_TREATMENT_STATE}_sq_lns_{SQLNS_STATE}_x_factor_{X_FACTOR_STATE}_diarrhea_{DIARRHEA_STATE}'
WASTING_TRANSITION_COUNT_COLUMN_TEMPLATE = '{WASTING_TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_sam_treatment_{SAM_TREATMENT_STATE}_mam_treatment_{MAM_TREATMENT_STATE}_sq_lns_{SQLNS_STATE}_x_factor_{X_FACTOR_STATE}_diarrhea_{DIARRHEA_STATE}'
STUNTING_STATE_PERSON_TIME_COLUMN_TEMPLATE = '{STUNTING_STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_sq_lns_{SQLNS_STATE}'
BIRTHS_COLUMN_TEMPLATE = '{BIRTH_METRIC}_in_{YEAR}_among_{SEX}_maternal_malnutrition_{MATERNAL_MALNUTRITION_STATE}_maternal_supplementation_{MATERNAL_SUPPLEMENTATION_TYPES}_itn_{ITN_STATE}'
DIARRHEAL_DISEASES_STATE_PERSON_TIME_COLUMN_TEMPLATE = '{DIARRHEAL_DISEASES_STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}_therapeutic_zinc_{THERAPEUTIC_ZINC_STATE}_preventative_zinc_{PREVENTATIVE_ZINC_STATE}'
DIARRHEAL_DISEASES_TRANSITION_COUNT_COLUMN_TEMPLATE = '{DIARRHEAL_DISEASES_TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}_therapeutic_zinc_{THERAPEUTIC_ZINC_STATE}_preventative_zinc_{PREVENTATIVE_ZINC_STATE}'


COLUMN_TEMPLATES = {
    'population': TOTAL_POPULATION_COLUMN_TEMPLATE,
    'deaths': DEATH_COLUMN_TEMPLATE,
    'ylls': YLLS_COLUMN_TEMPLATE,
    'ylds': YLDS_COLUMN_TEMPLATE,
    'disease_state_person_time': DISEASE_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'disease_transition_count': DISEASE_TRANSITION_COUNT_COLUMN_TEMPLATE,
    'wasting_state_person_time': WASTING_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'wasting_transition_count': WASTING_TRANSITION_COUNT_COLUMN_TEMPLATE,
    'stunting_state_person_time': STUNTING_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'births': BIRTHS_COLUMN_TEMPLATE,
    'diarrheal_diseases_state_person_time': DIARRHEAL_DISEASES_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'diarrheal_diseases_transition_count': DIARRHEAL_DISEASES_TRANSITION_COUNT_COLUMN_TEMPLATE,
}

NON_COUNT_TEMPLATES = [
]

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
YEARS = tuple(range(2022, 2027))
AGE_GROUPS = ('early_neonatal', 'late_neonatal', '1-5_months', '6-11_months', '12_to_23_months', '2_to_4')
STUNTING_STATES = ('cat4', 'cat3', 'cat2', 'cat1')
TREATMENT_STATES = ('covered', 'uncovered')
DICHOTOMOUS_RISK_STATES = ('cat2', 'cat1')
MATERNAL_SUPPLEMENTATION_TYPES = ('bep', 'mmn', 'ifa', 'uncovered')
BIRTH_METRICS = ('total_births', 'birth_weight_sum', 'low_weight_births')
CAUSES_OF_DEATH = (
    'other_causes',
    models.DIARRHEA.STATE_NAME,
    models.MEASLES.STATE_NAME,
    models.LRI.STATE_NAME,
    models.WASTING.MODERATE_STATE_NAME,
    models.WASTING.SEVERE_STATE_NAME,
)
CAUSES_OF_DISABILITY = (
    models.DIARRHEA.STATE_NAME,
    models.MEASLES.STATE_NAME,
    models.LRI.STATE_NAME,
    models.WASTING.MODERATE_STATE_NAME,
    models.WASTING.SEVERE_STATE_NAME,
)
DIARRHEAL_DISEASES_STATES = ('susceptible_to_diarrheal_diseases','diarrheal_diseases')
DIARRHEAL_DISEASES_TRANSITIONS = (
    'susceptible_to_diarrheal_diseases_to_diarrheal_diseases',
    'diarrheal_diseases_to_susceptible_to_diarrheal_diseases'
)

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_GROUP': AGE_GROUPS,
    'CAUSE_OF_DEATH': CAUSES_OF_DEATH,
    'CAUSE_OF_DISABILITY': CAUSES_OF_DISABILITY,
    'DISEASE_STATE_EXCL_DIARRHEA': [d for d in models.DISEASE_STATES if 'diarrheal_diseases' not in d],
    'DISEASE_TRANSITION_EXCL_DIARRHEA': [d for d in models.DISEASE_TRANSITIONS if 'diarrheal_diseases' not in d],
    'WASTING_STATE': models.WASTING.STATES,
    'WASTING_TRANSITION': models.WASTING.TRANSITIONS,
    'STUNTING_STATE': STUNTING_STATES,
    'SQLNS_STATE': TREATMENT_STATES,
    'SAM_TREATMENT_STATE': TREATMENT_STATES,
    'MAM_TREATMENT_STATE': TREATMENT_STATES,
    'X_FACTOR_STATE': DICHOTOMOUS_RISK_STATES,
    'BIRTH_METRIC': BIRTH_METRICS,
    'MATERNAL_MALNUTRITION_STATE': DICHOTOMOUS_RISK_STATES,
    'MATERNAL_SUPPLEMENTATION_TYPES': MATERNAL_SUPPLEMENTATION_TYPES,
    'ITN_STATE': TREATMENT_STATES,
    'DIARRHEAL_DISEASES_STATE': DIARRHEAL_DISEASES_STATES,
    'DIARRHEAL_DISEASES_TRANSITION': DIARRHEAL_DISEASES_TRANSITIONS,
    'DIARRHEA_STATE': DICHOTOMOUS_RISK_STATES,
    'THERAPEUTIC_ZINC_STATE': TREATMENT_STATES,
    'PREVENTATIVE_ZINC_STATE': TREATMENT_STATES,
}


LOW_BIRTH_WEIGHT_CUTOFF = 2500.0


# noinspection PyPep8Naming
def RESULT_COLUMNS(kind='all'):
    if kind not in COLUMN_TEMPLATES and kind != 'all':
        raise ValueError(f'Unknown result column type {kind}')
    columns = []
    if kind == 'all':
        for k in COLUMN_TEMPLATES:
            columns += RESULT_COLUMNS(k)
        columns = list(STANDARD_COLUMNS.values()) + columns
    else:
        template = COLUMN_TEMPLATES[kind]
        filtered_field_map = {field: values
                              for field, values in TEMPLATE_FIELD_MAP.items() if f'{{{field}}}' in template}
        fields, value_groups = filtered_field_map.keys(), itertools.product(*filtered_field_map.values())
        for value_group in value_groups:
            columns.append(template.format(**{field: value for field, value in zip(fields, value_group)}))
    return columns
