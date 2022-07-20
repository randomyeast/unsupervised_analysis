import torch

from lib.models.tvae import TVAE


if __name__ == "__main__":
    model_config = {
        'recurrent_dim': 128,
        'rnn_dim': 128,
        'num_layers': 3,
        'state_dim': 28,
        'z_dim': 32,
        'h_dim': 128,
        'final_hidden': True
    }

    # Test input
    states = torch.randn(60, 28, 28) # [seq_len, batch_size, state_dim]

    # Instantiate model
    model = TVAE(model_config)

    # Test forward pass of the model
    posterior = model(states)

    model.optimize(model.log.losses)
    print('something')