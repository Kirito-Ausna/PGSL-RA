_DATASET_DICT = {}

# Second level decorator
def register_dataset(name):  
    def decorator(cls):
        if name in _DATASET_DICT:
            raise ValueError("Cannot register duplicate config ({})".format(name))
        _DATASET_DICT[name] = cls
        return cls
    return decorator

# Get config by name
def get_dataset(name):
    if name not in _DATASET_DICT:
        raise ValueError("DATASET not found: {}".format(name))
    return _DATASET_DICT[name]