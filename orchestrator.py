import yaml
import importlib


class Orchestrator:
    def __init__(self, pipeline, first_input):
        self.config = self.load_config()
        self.registry = self.load_registry()  
        
        self.pipeline = self.init_actions(pipeline)  
        self.first_input = first_input 
        
    def load_registry(self):  # FIXED NAME
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
        import yaml
        with open("config.yaml") as f:
            return yaml.safe_load(f)    
    
    def validate_pipeline():
        return
        #check if all actions exist in registry and whether input output of the sequence is compatible
    
    def init_actions(self, pipeline_names):
        actions = []
        for name in pipeline_names:
            action_cls = self.registry[name]

            # Optional: Inject config if `__init__` takes one
            try:
                action = action_cls(config=self.config)
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
        
