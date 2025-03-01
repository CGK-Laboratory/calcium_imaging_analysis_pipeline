
import os
import json

def load_parameres(parameters_path=os.path.join(os.path.dirname(__file__), "default_parameter.json")):
    assert os.path.exists(parameters_path), "parameters file does not exist"

    with open(parameters_path) as file:
        default_parameters = json.load(file)

    return _global_parameters.update(default_parameters)

_global_parameters = {}

load_parameres()

def get_setting(key, default=None):
    """Retrieve the value of a setting."""
    return _global_parameters.get(key, default)

def set_setting(key, value):
    """Set the value of a setting."""
    _global_parameters[key] = value