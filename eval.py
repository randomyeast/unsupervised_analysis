import os
from re import I
import torch
import json
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.models import load_model
from lib.util.datasets import load_dataset

def evaluate(model, data_loader):
    """
    iteratively generates embeddings and reconstructions
    """
    reconstructions, embeddings = [], []
    for batch_idx, states in enumerate(tqdm(data_loader)):
        b_reconstructions, b_embeddings = model.reconstruct(states, embed=True)
        b_reconstructions = data_loader.dataset.postprocess(b_reconstructions.numpy())
        b_reconstructions = torch.tensor(b_reconstructions)
        reconstructions.append(b_reconstructions), embeddings.append(b_embeddings)

    return torch.cat(reconstructions, dim=0), torch.cat(embeddings, dim=0)

if __name__ == "__main__":
    # Parse evaluation arguments 
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

    # Instantiate dataset and data_loader
    print('-=-= Loading and preprocessing dataset =-=-')
    dataset = load_dataset(data_config)
    data_loader = DataLoader(
        dataset,
        batch_size=eval_config['batch_size'],
        shuffle=False,
        pin_memory=True,
    )

    # Instantiate model and training log
    model = load_model(model_config).to(device)
    model.eval()

    # Load in the best checkpoint
    model.load_best_checkpoint(args.config_dir)

    # Evaluate the model on the test set
    reconstructions, embeddings = evaluate(model, data_loader)
    
    # Save the reconstructions and embeddings
    torch.save(
        reconstructions,
        os.path.join(
            args.config_dir,
            'reconstructions',
            'reconstructions.pt'
    ))

    # Save the embeddings and embeddings
    torch.save(
        embeddings,
        os.path.join(
            args.config_dir,
            'embeddings',
            'embeddings.pt'
    ))