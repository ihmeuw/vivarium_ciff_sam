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
OUTPUT_SCENARIO_COLUMN = 'intervention.scenario'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
DEATH_COLUMN_TEMPLATE = 'death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
YLLS_COLUMN_TEMPLATE = 'ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
YLDS_COLUMN_TEMPLATE = 'ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
DISEASE_STATE_PERSON_TIME_COLUMN_TEMPLATE = '{DISEASE_STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
DISEASE_TRANSITION_COUNT_COLUMN_TEMPLATE = '{DISEASE_TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_state_{WASTING_STATE}'
WASTING_STATE_PERSON_TIME_COLUMN_TEMPLATE = '{WASTING_STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_treatment_{WASTING_TREATMENT_STATES}_sq_lns_{SQLNS_STATE}'
WASTING_TRANSITION_COUNT_COLUMN_TEMPLATE = '{WASTING_TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_wasting_treatment_{WASTING_TREATMENT_STATES}_sq_lns_{SQLNS_STATE}'
STUNTING_STATE_PERSON_TIME_COLUMN_TEMPLATE = '{STUNTING_STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_sq_lns_{SQLNS_STATE}'

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
}

NON_COUNT_TEMPLATES = [
]

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
YEARS = tuple(range(2022, 2027))
AGE_GROUPS = ('early_neonatal', 'late_neonatal', '1-5_months', '6-11_months', '12_to_23_months', '2_to_4')
STUNTING_STATES = ('cat4', 'cat3', 'cat2', 'cat1')
TREATMENT_STATES = ('covered', 'uncovered')
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

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_GROUP': AGE_GROUPS,
    'CAUSE_OF_DEATH': CAUSES_OF_DEATH,
    'CAUSE_OF_DISABILITY': CAUSES_OF_DISABILITY,
    'DISEASE_STATE': models.DISEASE_STATES,
    'DISEASE_TRANSITION': models.DISEASE_TRANSITIONS,
    'WASTING_STATE': models.WASTING.STATES,
    'WASTING_TRANSITION': models.WASTING.TRANSITIONS,
    'STUNTING_STATE': STUNTING_STATES,
    'SQLNS_STATE': TREATMENT_STATES,
    'WASTING_TREATMENT_STATE': TREATMENT_STATES,
}


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
