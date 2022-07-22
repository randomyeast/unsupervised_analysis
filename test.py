import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.models.tvae import TVAE
from lib.util.datasets.random_trajectories import RandomTrajectories 

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    model_config = {
        'recurrent_dim': 64,
        'rnn_dim': 64,
        'num_layers': 2,
        'state_dim': 2,
        'z_dim': 8,
        'h_dim': 64,
        'final_hidden': True,
    }

    # Create dataset and data loader
    data = RandomTrajectories() 
    data_loader = DataLoader(data, batch_size=32, shuffle=True)

    # Instantiate model
    model = TVAE(model_config)

    # Train the model
    for num_epochs in tqdm(range(10)):
        for batch in data_loader:
            _ = model(batch)
            reconstruction = model.reconstruct(batch) 
            model.optimize(model.log.losses)
        print(model.log)


    print('something')