import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """Base class for datasets of fixed-length trajectories.
    """
    def __init__(self):
        self.states = self.load_data()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]