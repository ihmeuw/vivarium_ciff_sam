from abc import abstractmethod, ABC
from typing import Dict, Tuple

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import get_exposure_post_processor
from vivarium_public_health.utilities import EntityString

from vivarium_ciff_sam.constants import data_keys


class LBWSGRisk(Risk, ABC):
    """"
    Risk component for the individual aspects of LBWSG (i.e. birth weight and gestational age).
    `risk_factor.low_birth_weight_and_short_gestation` must exist.
    """

    LBWSG_EXPOSURE_PIPELINE_NAME = f'{data_keys.LBWSG.name}.exposure'

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
            the type and name of a risk, specified as "type.name". Type is singular.
        """
        self.risk = EntityString(risk)
        self.configuration_defaults = {f'{self.risk.name}': Risk.configuration_defaults['risk']}
        self.exposure_distribution = None
        self._sub_components = []

    @property
    def propensity_column_name(self) -> str:
        return f'{self.risk.name}_propensity'

    @property
    def propensity_randomness_stream_name(self) -> str:
        return f'initial_{self.risk.name}_propensity'

    @property
    def propensity_pipeline_name(self) -> str:
        return f'{self.risk.name}.propensity'

    @property
    def exposure_pipeline_name(self) -> str:
        return f'{self.risk.name}.exposure'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = self.get_propensity_randomness_stream(builder)
        self.propensity = self.get_propensity_pipeline(builder)
        self.exposure = self.get_exposure_pipeline(builder)
        self.lbwsg_exposure = LBWSGRisk.get_lbwsg_exposure_pipeline(builder)
        self.category_endpoints = self.get_category_endpoints(builder)

        self.population_view = self.get_population_view(builder)
        self.register_simulant_initializer(builder)

    def get_propensity_randomness_stream(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(self.propensity_randomness_stream_name)

    def get_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: self.population_view.get(index)[self.propensity_column_name],
            requires_columns=[self.propensity_column_name]
        )

    def get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.get_current_exposure,
            requires_columns=['age', 'sex'],
            requires_values=[self.propensity_pipeline_name, LBWSGRisk.LBWSG_EXPOSURE_PIPELINE_NAME],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

    def get_category_endpoints(self, builder: Builder) -> Dict[str, Tuple[float, float]]:
        category_endpoints = {cat: self.parse_description(description)
                              for cat, description
                              in builder.data.load(f'risk_factor.{data_keys.LBWSG.name}.categories').items()}
        return category_endpoints

    def get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name])

    def register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.propensity_column_name],
            requires_streams=[self.propensity_randomness_stream_name]
        )

    def get_current_exposure(self, index):
        propensities = self.propensity(index)
        lbwsg_categories = self.lbwsg_exposure(index)

        def get_exposure_from_category(row: pd.Series) -> float:
            category_endpoints = self.category_endpoints[row[lbwsg_categories.name]]
            exposure = row[propensities.name] * (category_endpoints[1] - category_endpoints[0]) + category_endpoints[0]
            return exposure

        exposures = pd.concat([lbwsg_categories, propensities], axis=1).apply(get_exposure_from_category, axis=1)
        exposures.name = f'{self.risk}.exposure'
        return exposures

    @staticmethod
    def get_lbwsg_exposure_pipeline(builder: Builder) -> Pipeline:
        return builder.value.get_value(LBWSGRisk.LBWSG_EXPOSURE_PIPELINE_NAME)

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
