from re import I
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
import os

import matplotlib.pyplot as plt
from celluloid import Camera
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

BASE_PATH = os.path.join(os.getcwd(), 'lib/util/datasets/mouse_v1')
########### MOUSE DATASET FRAME WIDTH
# TODO: add these to config file
FRAME_WIDTH_TOP = 640
FRAME_HEIGHT_TOP = 480

def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[2] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.divide(data - shift, scale)

def normalize_video(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[2] // 1 
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.divide(data - shift, scale)

def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[1] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.multiply(data, scale) + shift


def transform_mars_to_svd_components(data,
                                center_index=3,
                                n_components=5,
                                svd_computer=None,
                                mean=None,
                                stack_agents=False,
                                stack_axis=1,
                                save_svd=False):

    # Grab trajectory length for reshaping at end
    traj_len = data.shape[1]

    # Concatenate all trajectories and reshape feature dim
    data = np.concatenate(data, axis=0)
    data = data.reshape(-1, 7, 2)

    # Center the data using given center_index
    mouse_center = data[:, center_index, :]
    centered_data = data - mouse_center[:, np.newaxis, :]

    # Rotate such that keypoints 3 and 6 are parallel with the y axis
    mouse_rotation = np.arctan2(
        data[:, 3, 0] - data[:, 6, 0], data[:, 3, 1] - data[:, 6, 1])

    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                   [np.sin(mouse_rotation), np.cos(mouse_rotation)]]).transpose((2, 0, 1)))

    # Encode mouse rotation as sine and cosine
    mouse_rotation = np.concatenate([np.sin(mouse_rotation)[:, np.newaxis], np.cos(
        mouse_rotation)[:, np.newaxis]], axis=-1)

    centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
    centered_data = centered_data.transpose((0, 2, 1))

    centered_data = centered_data.reshape((-1, 14))

    if mean is None:
        mean = np.mean(centered_data, axis=0)
    centered_data = centered_data - mean

    # Compute SVD components
    if svd_computer is None:
        svd_computer = TruncatedSVD(n_components=n_components)
        svd_data = svd_computer.fit_transform(centered_data)
    else:
        svd_data = svd_computer.transform(centered_data)
        explained_variances = np.var(svd_data, axis=0) / np.var(centered_data, axis=0).sum()

    # Concatenate state as mouse center, mouse rotation and svd components
    data = np.concatenate([mouse_center, mouse_rotation, svd_data], axis=1)

    if save_svd:
        with open(os.path.join(f'{BASE_PATH}/svd/svd.pickle'), 'wb') as f:
            pickle.dump(svd_computer, f)
        with open(os.path.join(f'{BASE_PATH}/svd/mean.pickle'), 'wb') as f:
            pickle.dump(mean, f)

    # Reshape to trajectory length -- this is the problem
    data = np.stack(np.vsplit(data, len(data)//traj_len))

    return data, svd_computer, mean


def unnormalize_keypoint_center_rotation(keypoints, center, rotation):

    keypoints = keypoints.reshape((-1, 7, 2))

    # Apply inverse rotation
    rotation = -1 * rotation
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation),  np.cos(rotation)]]).transpose((2, 0, 1))
    centered_data = np.matmul(R, keypoints.transpose(0, 2, 1))

    keypoints = centered_data + center[:, :, np.newaxis]
    keypoints = keypoints.transpose(0, 2, 1)

    return keypoints.reshape(-1, 14)


def transform_svd_to_keypoints(data, svd_computer, mean, stack_agents = False,
                            stack_axis = 0):

    # Save trajectory length for reshaping at end
    traj_len = data.shape[1]

    # Concatenate all trajectories
    data = np.concatenate(data, axis=0)

    # Grab individual svd components
    num_components = svd_computer.n_components
    center = data[:, :2]
    rotation = data[:, 2:4]
    components = data[:, 4:4+num_components]

    # Inverse SVD
    keypoints = svd_computer.inverse_transform(components)

    # Mean add
    if mean is not None:
        keypoints = keypoints + mean

    # Compute rotation angle from sine and cosine representation
    rotation = np.arctan2(
        rotation[:, 0], rotation[:, 1])

    # Undo center and rotation
    keypoints = unnormalize_keypoint_center_rotation(
        keypoints,
        center,
        rotation
    )

    # Reshape to trajectory length
    keypoints = np.stack(np.vsplit(keypoints, len(keypoints)//traj_len))
    
    return keypoints