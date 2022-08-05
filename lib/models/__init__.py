from .core import BaseSequentialModel
from .tvae import TVAE

model_dict = {
    'tvae' : TVAE
}

def load_model(model_config):
    model_name = model_config['model_type'].lower()

    if model_name in model_dict:
        return model_dict[model_name](model_config)
    else:
        raise NotImplementedError
