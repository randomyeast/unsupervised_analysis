import os
from re import I
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.models import load_model
from lib.util.datasets import load_dataset, dataset_dict

def plot_random_reconstructions(model, eval_config, data_config, num_reconstructions):
    # Load the dataset in
    dataset = load_dataset(data_config)
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=512
    )
   
    # Combine training and evaluation sets
    data_loader.dataset.eval() #combine()

    # Generate reconstructions for entire dataset 
    vid_recons = model.reconstruct(
        data_loader,
        num_samples=eval_config['num_samples']
    )

    # Cannot convert cuda tensor to numpy array
    vid_recons = vid_recons.cpu().numpy()

    # Post-processing of the reconstructions
    vid_recons = dataset.postprocess(vid_recons)

    # Iterate through and save plots
    for i in range(num_reconstructions):
        # Select a random trajectory
        rand_idx = np.random.randint(0, vid_recons.shape[0])

        # Get original and postprocess
        orig_traj = data_loader.dataset[rand_idx]
        orig_traj = dataset.postprocess(orig_traj)

        # Get reconstruction
        recon_traj = vid_recons[rand_idx]

        # Plot original and reconstruction
        dataset.plot_reconstruction(
            orig_traj,
            recon_traj,
            path=os.path.join(
                args.config_dir,
                'example_reconstructions',
                f'recon_{i}'
            )
        )

if __name__ == "__main__":
    # Parse training arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True,
                        help='path to the configuration json to use for training')
    parser.add_argument('--num_reconstructions', type=int, required=True,
                        help='number of example reconstructions to plot')
    args = parser.parse_args()

    # Load in configuration dictionary for training
    with open(os.path.join(args.config_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    model_config = config['model_config']
    eval_config = config['eval_config']
    data_config = config['data_config']
 
    device = torch.device(eval_config['device'])   # TODO: add this into the configuration

    # Instantiate model and load best checkpoint
    model = load_model(model_config).to(device)
    model.load_best_checkpoint(args.config_dir)

    # Train model
    plot_random_reconstructions(
        model, 
        eval_config,
        data_config,
        args.num_reconstructions
    )
