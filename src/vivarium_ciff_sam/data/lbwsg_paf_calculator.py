from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from vivarium import Artifact, InteractiveContext

from vivarium_ciff_sam.constants import data_keys, metadata


def get_pafs(config: Path, input_draw: int, random_seed: int, age_group_id: int) -> pd.DataFrame:

    sim = InteractiveContext(config, setup=False)

    artifact_path = sim.configuration.input_data.artifact_path
    artifact = Artifact(artifact_path)

    age_bins = artifact.load(data_keys.POPULATION.AGE_BINS).reset_index().set_index('age_group_id')
    age_start = age_bins.loc[age_group_id, 'age_start']
    age_end = age_bins.loc[age_group_id, 'age_end']

    year_start = 2019
    year_end = 2020

    sim.configuration.update({
        'input_data': {
            'input_draw_number': input_draw,
        },
        'randomness': {
            'random_seed': random_seed,
        },
        'population': {
            'age_start': age_start,
            'age_end': age_end
        }
    })
    sim.setup()

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


def write_pafs_to_hdf(config: str, output_dir: str, input_draw: str, random_seed: str):
    config = Path(config)
    output_dir = Path(output_dir)
    input_draw = int(input_draw)
    random_seed = int(random_seed)

    pafs = pd.concat([get_pafs(config, input_draw, random_seed, age_group_id)
                      for age_group_id in metadata.AGE_GROUP.GBD_2019_LBWSG_RELATIVE_RISK])

    pafs.to_hdf(output_dir / f'draw_{input_draw}.csv', 'paf')


if __name__ == "__main__":
    write_pafs_to_hdf(*sys.argv[1:])
