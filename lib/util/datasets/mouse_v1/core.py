import os
from re import I
import sys
import torch
import numpy as np
from scipy.signal import medfilt2d as median_filter

from lib.util.datasets import TrajectoryDataset
from lib.util.plotting import plotting
from lib.util.misc import immediate_sub_dirs, json_to_keypoints

from lib.util.datasets.mouse_v1.preprocess import *

class MouseV1Dataset(TrajectoryDataset):
    """
    Basic dataset object for handling SINGLE mouse MARS keypoint estimates
    """
    # Declare mean and svd as none to allow for preprocessing to be done
    _svd, _mean = None, None

    def __init__(self, data_config):
        """
        Parameters
        ========== 
        data_config: dict
            A dictionary containing the following dataset attributes:
                - `root_data_dir`: str
                    The root directory to look for MARS keypoint estimates.
                    See <add link> note on how to structure directories for
                    search to work properly
                - `dataset_type`: str
                    The type of dataset to load (e.g. 'mouse_v1') - used in
                    the `load_dataset` method <add link>
                - `traj_len`: int
                    The length of the trajectory to use for training i.e.
                    the number of frames to use in each video.
        """
        super().__init__(data_config)

    def load_data(self, data_config):
        """
        Loads the dataset from the specified root directory and convert videos
        into trajectories of length specified by ``data_config[traj_len]``.
        
        Notes
        =====
        - This method assumes that there are two sets of keypoint estimates
          in the ``.json`` files output by MARS, both representing the same
          mouse.

        Parameters
        ==========
        data_config: dict
            A dictionary containing the attributes explained in the
            constructor.

        Returns
        =======
        trajectories: np.array
            Array of shape ``[num_trajs, traj_len, n_features]`` of pose
            trajectories.
        """
        # Load the raw MARS keypoint estimates
        vid_dict = self.load_raw_mars_keypoints(
            data_config['root_data_dir']
        )

        # Convert the keypoint estimates to fixed length trajectories
        vid_dict = self.convert_to_trajectories(
            vid_dict,
            traj_len=data_config['traj_len']
        )

        # Combine all trajectories into big array
        trajectories = np.concatenate([vid for vid in vid_dict.values()])

        return trajectories 

    def preprocess(self, data_config, trajectories):
        """
        Notes
        =====
        Preprocesses single mouse trajectories using the following steps:

        - `Centroid alignment`:
            Aligns each trajectory such that the neck base of the mouse in 
            the first frame is aligned with the origin.  
        - `Rotational alignment`:
            Aligns each trajectory such that the vector formed by the neck
            base and the tail base is aligned with the y-axis. 
        - `Decomposition`:
            - Combines all trajectories into one pose matrix
            - Centers the poses such that the neck base is at the origin
              and stores the original coordinates of the neck base
            - Rotates the poses such that the vector formed by the neck base
              and the tail base is aligned with the y-axis and stores the
              sine and cosine of the rotation angle.
            - Performs SVD and for each aligned pose, stores the weights
              associated with the right singular vectors needed to reconstruct
              the pose.
            - Concatenates the extracted centroid position, angle of rotation
              and the weights associated with the right singular vectors into
              a single matrix.
            - Splits the resulting matrix back up into its original
              trajectories and returns.
 
        Parameters
        ==========
        data_config: dict
            A dictionary containing the attributes explained in the \
            constructor
        trajectories: np.array
            Array of shape ``[num_traj, traj_len, n_features]`` pose
            trajectories to preprocess

        Returns
        =======
        trajectories: np.array
            Array of shape ``[num_traj, traj_len, n_features]`` of
            centroid and body-angle registered pose trajectories

        """
        
        # If preprocessing a single pose
        if len(trajectories.shape ) < 3:
            data = np.expand_dims(data, axis=0)
        
        # Register the pose trajectories w.r.t. centroid and body angle
        trajectories = self.register_pose_trajectories(
            trajectories,
            traj_len=data_config['traj_len']
        )

        # Filter and normalize the pose estimates
        trajectories = self.med_filter(trajectories)

        # Learn the SVD of the combined videos --- PROBLEM IS HERE
        data, self._svd, self._mean = transform_mars_to_svd_components(
                trajectories, 
                svd_computer=self._svd,
                mean=self._mean,
                save_svd=True
        )

        return data

    def register_pose_trajectories(self, trajectories, traj_len=61):
        """
        Register the pose trajectories w.r.t. centroid and body angle
        this function calls ``register_mars_trajectory_centroids`` and
        ``register_mars_trajectory_body_angles``.

        Parameters
        ==========
        trajectories: np.array 
            Array of shape ``[num_trajs, traj_len, num_feats]`` of pose
            trajectories to register
        """
        # register the pose centroids on the starting frame
        trajectories = self.register_mars_trajectory_centroids(
            trajectories,
            traj_len=traj_len
        )

        # register the pose angles on the starting frame
        trajectories = self.register_mars_trajectory_angles(
            trajectories, 
            traj_len=traj_len
        )
  
        return trajectories 

    def postprocess(self, trajectories):
        """
        Takes one or more preprocessed trajectories and reconstructs
        them from the decomposed pose and singular vectors.

        Parameters
        ==========
        trajectories: np.array
            Array of shape ``[num_trajs, traj_len, n_features]`` or 
            ``[traj_len, n_features]`` of pose trajectories to reconstruct

        Returns
        =======
        reconstructed_trajectories: np.array
            Array of shape ``[num_trajs, traj_len, n_features]`` or 
            ``[traj_len, n_features]`` of reconstructed pose trajectories

        """
        if len(trajectories.shape) < 3:
            trajectories = np.expand_dims(trajectories, axis=0)

        assert self._svd is not None, "Must have fit SVD to postprocess"
        assert self._mean is not None, "Must have stored unprocessed data mean to postprocess"

        trajectories = transform_svd_to_keypoints(
            trajectories,
            svd_computer =self._svd,
            mean = self._mean
        )

        return trajectories.squeeze()


    def med_filter(self, trajectories, kernel_size=7):
        """
        Applies a median filter over time to the trajectories to remove
        noise in the pose estimates.

        .. image:: images/median_filter.gif
        :align: center


        Parameters
        ==========
        trajectories: np.array
            Array of shape ``[num_trajs, traj_len, n_features]`` of pose
            trajectories to filter

        Returns
        =======
        filtered_trajectories: np.array
            Array of shape ``[num_trajs, traj_len, n_features]`` of filtered
            pose trajectories
        """
        trajectories = np.stack([median_filter(v, kernel_size=[kernel_size,1]) for v in trajectories])

        return trajectories 


    def register_mars_trajectory_angles(self, trajectories, traj_len=61):
        """Register the angle formed by the body vector and the
           y-axis on the starting frame

        .. image:: images/rotational_alignment.png
        :align: center

        Parameters
        ==========
        trajectories: np.array
            Array of shape (``[num_traj, traj_len, n_features]``) pose 
            trajectories to register.
        """
        # Compute the angle of rotation
        angle = np.arctan2(
            trajectories[:,0,6]-trajectories[:,0,12],
            trajectories[:,0,7]-trajectories[:,0,13]
        )

        # Compute the rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]).transpose(2,0,1)

        R = np.expand_dims(R, axis=1)
        R = np.repeat(R, traj_len, axis=1)

        # Apply the rotation matrix to the trajectories
        trajectories = trajectories.reshape(trajectories.shape[0], trajectories.shape[1], 7, 2).transpose(0,1,3,2)
        aligned = np.matmul(R, trajectories).transpose(0,1,3,2)
        aligned = aligned.reshape(aligned.shape[0], aligned.shape[1], -1)

        return aligned 


    def register_mars_trajectory_centroids(self, trajectories, traj_len=61):
        """Register the pose centroids on the starting frame

        .. image:: images/centroid_alignment.png
        :align: center

        Parameters
        ==========
        trajectories: np.array
            Array of shape (``[num_traj, traj_len, n_features]``) pose 
            trajectories to register.
        """

        # Get x position of the centroid for the first frame
        x_centr = np.expand_dims(trajectories[:,0,6], axis=1)
        x_centr = np.repeat(x_centr, 7, axis=1)
        x_centr = np.expand_dims(x_centr, axis=1)
        x_centr = np.repeat(x_centr, traj_len, axis=1)

        # Get y position of the centroid for the first frame
        y_centr = np.expand_dims(trajectories[:,0,7], axis=1)
        y_centr = np.repeat(y_centr, 7, axis=1)
        y_centr = np.expand_dims(y_centr, axis=1)
        y_centr = np.repeat(y_centr, traj_len, axis=1)

        # Subtract out first frame centroids for all trajectories
        trajectories[:,:,::2] = trajectories[:,:,::2] - x_centr
        trajectories[:,:,1::2] = trajectories[:,:,1::2] - y_centr

        return trajectories 

        
    def convert_to_trajectories(self, vid_dict, traj_len=61, sliding_window=1):
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
 
    def load_raw_mars_keypoints(self, root_data_dir):
        # TODO: add option for subset of videos
        videos = immediate_sub_dirs(root_data_dir)
        vid_dict = {}
        for video in videos:
            vid_name = video.split('/')[-2]
            pose_path = os.path.join(video, f'{vid_name}_pose_top_v1_8.json')
            vid_dict[vid_name] = json_to_keypoints(pose_path)

        return vid_dict

    
    @staticmethod
    def plot_trajectory(seq, path='./gifs/traj.gif'):
        """
        Plots a single mouse trajectory.
        """
        plotting.plot_mouse_sequence(seq, path=path)

    @staticmethod
    def plot_reconstruction(orig, recon, path='./gifs/recon.gif'):
        """
        Plot original and reconstructed trajectories
        on the same plot 
        """
        plotting.plot_mouse_reconstruction(orig, recon, path=path)