from .core import TrajectoryDataset
from .mouse_v1 import MouseV1Dataset

dataset_dict = {
    'mouse_v1' : MouseV1Dataset 
}

def load_dataset(dataset_config):
    dataset_name = dataset_config['dataset_type'].lower()

    if dataset_name in dataset_dict:
        return dataset_dict[dataset_name](dataset_config)
    else:
        raise NotImplementedError