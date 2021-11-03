from vivarium import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView
from vivarium_public_health.utilities import EntityString, TargetString


class DiseaseEffect:
    """A component to set a value based on a state in a cause model
    """

    configuration_defaults = {
        'effect_of_cause_on_target': {
            'measure': {
                'state': None,
            }
        }
    }

    def __init__(self, cause: str, target: str):
        self.cause = EntityString(cause)
        self.target = TargetString(target)
        self.configuration_defaults = {
            f'effect_of_{self.cause.name}_on_{self.target.name}': {
                self.target.measure: DiseaseEffect.configuration_defaults['effect_of_cause_on_target']['measure']
            }
        }

    @property
    def name(self):
        return f'risk_effect.{self.cause}.{self.target}'

    @property
    def initial_state_column_name(self):
        return f'initial_{self.cause.name}'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.effect_of_cause_on_target = self.get_effect_of_cause_on_target(builder)
        self.register_target_modifier(builder)
        self.population_view = self.get_population_view(builder)

    def get_effect_of_cause_on_target(self, builder: Builder) -> ConfigTree:
        effect_block = builder.configuration[f'effect_of_{self.cause.name}_on_{self.target.name}'][self.target.measure]
        return effect_block

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(f'{self.target.type}.{self.target.name}.{self.target.measure}',
                                              modifier=self.adjust_target,
                                              requires_columns=['age', 'sex', f'initial_{self.cause.name}'])

    def get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(self.initial_state_column_name)

    def adjust_target(self, index, target):
        pop = self.population_view.subview([self.initial_state_column_name]).get(index)
        for state, value in self.effect_of_cause_on_target.to_dict().items():
            target[pop[self.initial_state_column_name] == state] = value
        return target

    def __repr__(self):
        return f"RiskEffect(risk={self.cause}, target={self.target})"
