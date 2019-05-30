import json
from collections import defaultdict
from .files import load_config_from_json

class Config(object):
    def __init__(self, name, values):
        def _none_fac():
            return None
        self.name = name
        self.values = defaultdict(_none_fac, **values)

    @classmethod
    def from_json(cls, json_filename):
        model_conf, optim_conf = load_config_from_json(json_filename)

        model_name = model_conf['name']
        del model_conf['name']
        model_config = Config(model_name, model_conf)

        optim_name = optim_conf['name']
        del optim_conf['name']
        optim_config = Config(optim_name, optim_conf)
        return model_config, optim_config

    def __getattr__(self, key):
        return self.values[key]

    def __setattr__(self, key, value):
        if key in ('name', 'values'):
            self.__dict__[key] = value
        else:
            self.values[key] = value
