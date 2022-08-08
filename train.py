import os
import torch
import json
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.models import load_model
from lib.util.datasets import load_dataset


def train(model, data_loader, train_config, writer):
    for stage_num, stage_epochs in enumerate(train_config['num_epochs']):
        # Reset stage parameters
        model.stage = stage_num
        best_val_log = {}

        print(f'-=-= Training stage {stage_num} =-=-')
        for epoch in tqdm(range(stage_epochs)):
            # Run a single epoch of training and print out the loss
            epoch_train_log = model.fit(data_loader)
            writer = epoch_train_log.write_to_tensorboard(writer, epoch, train=True)
            print(f'-=-= Epoch {epoch} =-=-')
            print(epoch_train_log) 

            # Run evaluation every checkpoint_frequency epochs 
            if epoch % train_config['checkpoint_frequency'] == 0:
                epoch_val_log = model.test(data_loader)
                writer = epoch_val_log.write_to_tensorboard(writer, epoch, train=False)
                model.save_checkpoint(args.config_dir, name=f'ckpt_{epoch}')

                # Check to see if current model is best
                if epoch == 0 or sum(epoch_val_log.losses.values()) < \
                                 sum(best_val_log.losses.values()):

                    print(f'-=-= Saving best model for stage {model.stage} =-=-') 
                    best_val_log = epoch_val_log   
                    model.save_checkpoint(args.config_dir, name=f'best_{epoch}')

    return writer

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
    train_config = config['train_config']
    data_config = config['data_config']
 
    device = torch.device(train_config['device'])   # TODO: add this into the configuration

    # Instantiate dataset and data_loader
    print('-=-= Loading and preprocessing dataset =-=-')
    dataset = load_dataset(data_config)
    data_loader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        pin_memory=True,
    )

    # Instantiate model and training log
    model = load_model(model_config).to(device)
    logger = []

    # Create tensorboard writer for logging
    writer = SummaryWriter(os.path.join(args.config_dir, 'log'))

    # Create directory for saving checkpoints at each stage
    for stage_num, _ in enumerate(train_config['num_epochs']):
        stage_dir = os.path.join(args.config_dir, 'checkpoints', f'stage_{stage_num}')
        if not os.path.exists(stage_dir):
            os.mkdir(stage_dir)

    # Train model
    writer = train(model, data_loader, train_config, writer)
    writer.close()