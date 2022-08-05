import torch
import numpy as np
from torch.utils.data import Dataset

TRAIN = 1
EVAL = 2

class TrajectoryDataset(Dataset):
    """Base class for datasets of fixed-length trajectories.
    """
    mode = TRAIN
    test_proportion = 0.2

    def __init__(self, data_config, all_states=None):
        """Abstract dataset class for fixed-length trajectory datasets. When
        working with any new type of dataset, create a new subclass of this
        class.
        
      
        Parameters
        ----------
        data_config: dict
            A dictionary containing the following model attributes:
                - `test_proportion`: float (optional)
                    Proportion of dataset to use for testing.
                - `traj_len`: int (optional)
                    Length of trajectories in frames to use. Not used in this
                    object but it is recommended to use this in the
                    implementation of the subclass.
        """
 
        # Submodule's load_data method must return all states
        if all_states is None: # In case you want to pass data directly into the object
            trajectories = self.load_data(data_config)
            trajectories = self.preprocess(
                data_config,
                trajectories
            )
            if not torch.is_tensor(trajectories):
                trajectories = torch.tensor(trajectories).float()

        # Split dataset into training and testing set
        if hasattr(data_config, 'test_proportion'):
            self.train_states, self.test_states = self._split_dataset(
                trajectories,
                data_config['test_proportion']
            )
        else:
            self.train_states, self.test_states = self._split_dataset(
                trajectories 
            )

        # Add dictionary for switching training and testing set 
        self.states = {
            TRAIN: self.train_states,
            EVAL: self.test_states
        }

    def _split_dataset(self, all_states, test_proportion=0.2):
        """Split the dataset into training and testing sets
        
        Parameters
        ----------
        self.test_proportion : float 
            Proportion of dataset to use for testing. This is not passed as an
            argument, but is set by default to be 0.2.
        """
        test_len = int(len(all_states) * test_proportion)
        train_set, test_set = torch.utils.data.random_split(
            all_states,
            [len(all_states)-test_len, test_len],
            generator=torch.Generator()
        )
        
        return train_set, test_set

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.states[self.mode])

    def __getitem__(self, idx):
        """Return a fixed length trajectory from either training or testing set."""
        return self.states[self.mode][idx] 

    def train(self):
        """Set the dataset to training mode."""
        self.mode = TRAIN
    
    def eval(self):
        """Set the dataset to evaluation mode."""
        self.mode = EVAL 

    def load_data(self, data_config):
        """
        Load the dataset, this must be implemented in the subclass
        used for training the model. It must return a tensor of shape
        ``[num_trajectories, trajectory_length, feature_dim]``
        """
        raise NotImplementedError("Subclass must implement load_data")

    def fit_preprocess(self):
        """
        Fit parameters used for preprocessing e.g. singular value 
        decomposition
        """
        raise NotImplementedError("Subclass must implement fit_preprocess")