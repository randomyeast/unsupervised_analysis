import torch
import numpy as np

from lib.util.datasets import TrajectoryDataset

class RandomTrajectories(TrajectoryDataset):
    
    def load_data(self):
        states = self.generate_data()
        return torch.tensor(states).float()
    
    def generate_data(self, traj_len=10, num_trajs=1000):
        trajs = []
        while len(trajs) < num_trajs:
            traj = []
            while len(traj) < traj_len:
                traj.append(next(self.generate_trajectory()))
            trajs.append(np.stack(traj))
        return np.stack(trajs)

    def generate_trajectory(self):
        traj = np.array([0.,0.])
        while True:
            traj += np.random.randn(2)
            yield traj 