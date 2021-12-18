from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from vivarium import Artifact, InteractiveContext
from vivarium_public_health.population import BasePopulation
from vivarium_public_health.risks import Risk

from vivarium_ciff_sam.components import LowBirthWeight, ShortGestation
from vivarium_ciff_sam.constants import data_keys, metadata
from vivarium_ciff_sam.paths import ARTIFACT_ROOT


def get_pafs(input_draw: int, random_seed: int, age_group_id: int, population_size: int) -> pd.DataFrame:

    location = metadata.LOCATIONS[0]
    artifact_path = ARTIFACT_ROOT / f'{location.lower()}.hdf'
    artifact = Artifact(artifact_path)

    age_bins = artifact.load(data_keys.POPULATION.AGE_BINS).reset_index().set_index('age_group_id')
    age_start = age_bins.loc[age_group_id, 'age_start']
    age_end = age_bins.loc[age_group_id, 'age_end']
    year_start = 2019
    year_end = 2020

    components = [
        BasePopulation(),
        Risk('risk_factor.low_birth_weight_and_short_gestation'),
        LowBirthWeight(),
        ShortGestation(),
    ]

    configuration = {
        'input_data': {
            'input_draw_number': input_draw,
            'location': location,
            'artifact_path': artifact_path,
        },
        'interpolation': {
            'order': 0,
            'extrapolate': True,
        },
        'randomness': {
            'map_size': 1_000_000,
            'key_columns': ['entrance_time', 'age'],
            'random_seed': random_seed,
        },
        'time': {
            'start': {
                'year': 2022,
                'month': 1,
                'day': 1,
            },
            'end': {
                'year': 2026,
                'month': 12,
                'day': 31,
            },
            'step_size': 0.5
        },
        'population': {
            'population_size': population_size,
            'age_start': age_start,
            'age_end': age_end
        }
    }

    sim = InteractiveContext(components=components, configuration=configuration)

    pop = sim.get_population()
    gestational_ages = sim.get_value('short_gestation.exposure')(pop.index)
    birth_weights = sim.get_value('low_birth_weight.exposure')(pop.index)

    interpolators = artifact.load(data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR)

    def calculate_paf_by_sex(sex: str) -> float:
        sex_mask = pop['sex'] == sex
        interpolator = pickle.loads(bytes.fromhex(
            interpolators.loc[(sex, age_start, age_end, year_start, year_end,
                               'diarrheal_diseases', 'excess_mortality_rate'),
                              f'draw_{input_draw}']
        ))
        rrs = np.exp(interpolator(gestational_ages[sex_mask], birth_weights[sex_mask], grid=False))
        mean_rr = rrs.mean()
        paf = (mean_rr - 1) / mean_rr
        return paf

    pafs = pd.DataFrame([{'sex': sex,
                          'age_start': age_start,
                          'age_end': age_end,
                          'year_start': year_start,
                          'year_end': year_end,
                          f'draw_{input_draw}': calculate_paf_by_sex(sex)}
                         for sex in ['Female', 'Male']])
    return pafs


def write_pafs_to_hdf(output_dir: str, input_draw: str, random_seed: str, population_size: str = '200000'):
    output_dir = Path(output_dir)
    input_draw = int(input_draw)
    random_seed = int(random_seed)
    population_size = int(population_size)

    pafs = pd.concat([get_pafs(input_draw, random_seed, age_group_id, population_size)
                      for age_group_id in metadata.AGE_GROUP.GBD_2019_LBWSG_RELATIVE_RISK])

    pafs.to_hdf(output_dir / f'draw_{input_draw}.csv', 'paf')


if __name__ == "__main__":
    write_pafs_to_hdf(*sys.argv[1:])
