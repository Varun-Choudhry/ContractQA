import yaml
import importlib
import os
class Orchestrator:
    _config = None  
    _config_path = None

    
    def __init__(self, pipeline, first_input):
        self.registry = self.load_registry()
        self.pipeline = self.init_actions(pipeline)
        self.first_input = first_input

    def load_registry(self): 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        registry_path = os.path.join(base_dir, 'registry.yaml')

        with open(registry_path, 'r') as file:
            raw_registry = yaml.safe_load(file)

        registry = {}
        for name, info in raw_registry.items():
            module_path, class_name = info['class_path'].rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            registry[name] = cls
        return registry
    
    @classmethod
    def set_config(cls, path="config.yaml"):
        """Set the global config path and load config."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r") as f:
            cls._config = yaml.safe_load(f)
        cls._config_path = path
        

    @classmethod
    def get_config(cls):
        if cls._config is None:
            raise RuntimeError
        return cls._config

    def init_actions(self, pipeline_steps):
        actions = []
        for step in pipeline_steps:
            action_key, mode = step.split(":")
            action_cls = self.registry[action_key]

            try:
                action = action_cls(config=self.get_config(), mode=mode)
            except TypeError:
                action = action_cls()

            actions.append(action)
        return actions

    def run(self):
        input_data = self.first_input
        for action in self.pipeline:
            input_model = action.InputSchema(**input_data)
            output_model = action.execute(input_model)
            input_data = output_model.model_dump()
        return input_data
