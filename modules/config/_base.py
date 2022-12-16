_CONFIGS_DICT = {}

# Second level decorator
def register_config(name):  
    def decorator(cls):
        if name in _CONFIGS_DICT:
            raise ValueError("Cannot register duplicate config ({})".format(name))
        _CONFIGS_DICT[name] = cls
        return cls
    return decorator

# Get config by name
def get_config(name):
    if name not in _CONFIGS_DICT:
        raise ValueError("Config not found: {}".format(name))
    return _CONFIGS_DICT[name]