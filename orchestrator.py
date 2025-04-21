import yaml
import importlib

class Orchestrator:
    def __init__(self, pipeline, first_input):
        self.config = self.load_config()
        self.registry = self.load_registry()
        self.pipeline = self.init_actions(pipeline)
        self.first_input = first_input

    def load_registry(self): 
        with open('registry.yaml', 'r') as file:
            raw_registry = yaml.safe_load(file)

        registry = {}
        for name, info in raw_registry.items():
            module_path, class_name = info['class_path'].rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            registry[name] = cls
        return registry

    def load_config(self):
        with open("config.yaml") as f:
            return yaml.safe_load(f)  

    def init_actions(self, pipeline_steps):
        actions = []
        for step in pipeline_steps:
            action_key, mode = step.split(":")
            #action_key=convert_document_action
            #mode=semantic
            action_cls = self.registry[action_key]

            try:
                action = action_cls(config=self.config, mode=mode)
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
