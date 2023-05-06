_PIPLINE_DICT = {}

# Second level decorator
def register_pipeline(name):
    def decorator(cls):
        if name in _PIPLINE_DICT:
            raise ValueError("Cannot register duplicate config ({})".format(name))
        _PIPLINE_DICT[name] = cls
        return cls
    return decorator

# Get the pipeline by name
def get_pipeline(name):
    if name not in _PIPLINE_DICT:
        raise ValueError("Pipeline not found: {}".format(name))
    return _PIPLINE_DICT[name]