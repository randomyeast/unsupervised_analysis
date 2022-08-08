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

def embed(model, eval_config, data_config):
    # Instantiation of the dataset should be instantiation of the base class
    dataset_type = dataset_dict[data_config['dataset_type']]
    vid_dict = dataset_type.load_vid_dict(eval_config['root_data_dir'])
    vid_dict = dataset_type.convert_to_trajectories(
        vid_dict,
        traj_len=data_config['traj_len']
    )

    # Iterate through the videos and embed     
    for vid_name, vid_data in vid_dict.items():
        print(f"-=-= Generating embeddings for {vid_name} =-=-")

        # Create dataset using states from current video
        dataset = dataset_type(
            data_config,
            trajectories=vid_data
        )
        data_loader = DataLoader(
            dataset,
            batch_size=eval_config['batch_size'],
            shuffle=False
        )

        # Embed the entire video
        data_loader.dataset.combine()

        # Generate embeddings
        vid_zs = model.embed(data_loader)

        # Cannot convert cuda tensor to numpy array
        vid_zs = vid_zs.cpu()

        # Save embeddings to name specified in eval_config
        save_path = os.path.join(
            eval_config['root_data_dir'],
            vid_name,
            f'{eval_config["save_name"]}_embeddings'
        )
        np.save(save_path, vid_zs.numpy())


if __name__ == "__main__":
    # Parse training arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True,
                        help='path to the configuration json to use for training')
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

    # Generate and save embeddings 
    embed(model, eval_config, data_config)