import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

from lib.util.misc import immediate_sub_dirs

TRAIN = 1
EVAL = 2
COMBINED = 3

class TrajectoryDataset(Dataset):
    """Base class for datasets of fixed-length trajectories.
    """
    mode = TRAIN
    test_proportion = 0.2

    def __init__(self, data_config, trajectories=None, preprocess=True):
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
        if trajectories is None: # In case you want to pass data directly into the object
            trajectories = self.load_data(data_config)
        
        # If user wants to preprocess the data
        if preprocess:
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
            EVAL: self.test_states,
            COMBINED: ConcatDataset((self.train_states, self.test_states)) 
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

    def combine(self):
        """
        Combine the training and testing sets into a single dataset for
        embedding
        """
        self.mode = COMBINED

    def load_data(self, data_config):
        """
        Load the dataset, this must be implemented in the subclass
        used for training the model. It must return a tensor of shape
        ``[num_trajectories, trajectory_length, feature_dim]``
        """
        raise NotImplementedError("Subclass must implement load_data")

    def load_vid_dict(dataset_type, root_data_dir):
        # TODO: add option for subset of videos
        videos = immediate_sub_dirs(root_data_dir)
        vid_dict = {}
        for video in videos:
            vid_name = video.split('/')[-2]
            pose_path = os.path.join(video, f'{vid_name}_pose_top_v1_8.json')
            vid_dict[vid_name] = dataset_type.load_video(pose_path) #json_to_keypoints(pose_path)

        return vid_dict

    @staticmethod
    def load_video(path):
        raise NotImplementedError("Subclass must implement load_video")

    @staticmethod
    def convert_to_trajectories(vid_dict, traj_len=61, sliding_window=1):
        for (video_name, pre_pose) in vid_dict.items():

            pre_pose = pre_pose[:,0,:,:]
            pre_pose = pre_pose.transpose(0,2,1)
            pre_pose = pre_pose.reshape(pre_pose.shape[0], -1) 

            # Pads the beginning and end of the sequence with duplicate frames
            pad_vec = np.pad(
                pre_pose, 
                ((traj_len//2, traj_len-1-traj_len//2), (0, 0)),
                mode='edge'
            )

            trajectories = np.stack([
                pad_vec[i:len(pad_vec)+i-traj_len+1:sliding_window] for i in range(traj_len)
            ], axis=1)

            vid_dict[video_name] = trajectories

        return vid_dict