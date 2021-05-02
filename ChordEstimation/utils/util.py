import os

def get_instance(module, name, config, *args, **kwargs):
    """Get instance from a certain module using the args and kwargs
    """
    return getattr(module, config[name]['name'])(*args, **kwargs, **config[name]['args'])
