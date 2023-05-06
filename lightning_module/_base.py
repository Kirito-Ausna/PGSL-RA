_TASK_DICT = {}

# Second level decorator
def register_task(name):  
    def decorator(cls):
        if name in _TASK_DICT:
            raise ValueError("Cannot register duplicate config ({})".format(name))
        _TASK_DICT[name] = cls
        return cls
    return decorator

# Get config by name
def get_task(name):
    if name not in _TASK_DICT:
        raise ValueError("DATASET not found: {}".format(name))
    return _TASK_DICT[name]