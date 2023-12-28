from pathlib import Path

import vivarium_ciff_sam
from vivarium_ciff_sam.constants import metadata

BASE_DIR = Path(vivarium_ciff_sam.__file__).resolve().parent

ARTIFACT_ROOT = BASE_DIR / 'artifacts'
MODEL_SPEC_DIR = BASE_DIR / 'model_specifications'

TEMPORARY_PAF_DIR = ARTIFACT_ROOT / 'temporary_pafs'
