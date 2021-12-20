from abc import abstractmethod, ABC
from typing import Dict, List, Tuple

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import get_exposure_post_processor
from vivarium_public_health.risks.distributions import SimulationDistribution

from vivarium_ciff_sam.constants import data_keys


class LBWSGRisk(Risk, ABC):
    """"
    Risk component for the individual aspects of LBWSG (i.e. birth weight and gestational age).
    `risk_factor.low_birth_weight_and_short_gestation` must exist.
    """

    def __init__(self, risk: str):
        super(LBWSGRisk, self).__init__(risk)
        self.lbwsg_exposure_pipeline_name = f'{data_keys.LBWSG.name}.exposure'

    ##########################
    # Initialization methods #
    ##########################

    def _get_exposure_distribution(self) -> SimulationDistribution:
        return None

    ##############
    # Properties #
    ##############

    @property
    def sub_components(self) -> List:
        return []

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        super().setup(builder)
        self.lbwsg_exposure = self._get_lbwsg_exposure_pipeline(builder)
        self.category_endpoints = self._get_category_endpoints(builder)

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self._get_current_exposure,
            requires_columns=['age', 'sex'],
            requires_values=[self.propensity_pipeline_name, self.lbwsg_exposure_pipeline_name],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

    def _get_lbwsg_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.get_value(self.lbwsg_exposure_pipeline_name)

    def _get_category_endpoints(self, builder: Builder) -> Dict[str, Tuple[float, float]]:
        category_endpoints = {cat: self.parse_description(description)
                              for cat, description
                              in builder.data.load(f'risk_factor.{data_keys.LBWSG.name}.categories').items()}
        return category_endpoints

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index):
        propensities = self.propensity(index)
        lbwsg_categories = self.lbwsg_exposure(index)

        def get_exposure_from_category(row: pd.Series) -> float:
            category_endpoints = self.category_endpoints[row[lbwsg_categories.name]]
            exposure = row[propensities.name] * (category_endpoints[1] - category_endpoints[0]) + category_endpoints[0]
            return exposure

        exposures = pd.concat([lbwsg_categories, propensities], axis=1).apply(get_exposure_from_category, axis=1)
        exposures.name = f'{self.risk}.exposure'
        return exposures

    ##################
    # Helper methods #
    ##################

    @staticmethod
    @abstractmethod
    def parse_description(description: str) -> Tuple[float, float]:
        # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'
        return 0.0, 1.0


class LowBirthWeight(LBWSGRisk):

    def __init__(self):
        super().__init__('risk_factor.low_birth_weight')

    @staticmethod
    def parse_description(description: str) -> Tuple[float, float]:
        # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'
        endpoints = tuple(float(val) for val in description.split(', [')[1].split(')')[0].split(', '))
        return endpoints


class ShortGestation(LBWSGRisk):

    def __init__(self):
        super().__init__('risk_factor.short_gestation')

    @staticmethod
    def parse_description(description: str) -> Tuple[float, float]:
        # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'
        endpoints = tuple(float(val) for val in description.split('- [')[1].split(')')[0].split(', '))
        return endpoints
