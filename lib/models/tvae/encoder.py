import torch
import torch.nn as nn

from lib.distributions import Normal

class TVAEEncoder(nn.Module):
    r"""Detailed explanation available :doc:`here <tvae_encoder_explanation>`"""
    # TODO: Define the default model parameters
    
    def __init__(self, log, **kwargs):
        r"""Defines the encoder network architecture

        Parameters
        ========== 
        kwargs: dict
            A dictionary containing the following model attributes:
                - `state_dim`: int
                    The dimensionality of the input space i.e. the number of 
                    elements in each input state vector, and the number of 
                    columns in :math:`W_{0}`.
                - `rnn_dim`: int
                    The dimensionality of the hidden state of the GRU. This
                    corresponds to the number of rows in :math:`W_{0}` and the 
                    dimensionality of the hidden states :math:`h_{t,j}`.
                - `num_layers`: int
                    The number of layers :math:`M` to use in the recurrent
                    portion of the encoder.
                - `h_dim`: int
                    The dimensionality of the space mapped to by the first
                    two fully-connected layers in the encoder ``enc_fc``
                - `z_dim`: int
                    The dimensionality of the latent space mapped to by the
                    separate fully-connected layers in the encoder
                    ``enc_mean`` and ``enc_logvar``.
                - `final_hidden`: bool, optional
                    If ``True``, the final hidden state of the RNN is used as
                    described in the second model variation above. If omitted,
                    this is assumed to be ``False``.
        """

        super(TVAEEncoder, self).__init__()

        # Update the attributes with the passed in kwargs
        self.__dict__.update(kwargs)

        # Make log available to encoder
        self.log = log

        # Define recurrent portion of the encoder
        self.enc_birnn = nn.GRU(
            self.state_dim*2,           # We always concatenate the states and actions
            self.rnn_dim,               
            num_layers=self.num_layers,
            bidirectional=True
        )

        # TODO: Add variant that uses output of last recurrent unit

        # Define the fully connected portion of the encoder
        self.enc_fc = nn.Sequential(
            nn.Linear(2*self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

        # Define the module learning latent mean ...
        self.enc_mean = nn.Linear(
            self.h_dim,
            self.z_dim
        )
        # ... and log variance
        self.enc_logvar = nn.Linear(
            self.h_dim,
            self.z_dim
        )

    def forward(self, states, actions=None):
        """Computes the mean and log variance of the posterior distribution

        Parameters
        ----------
        states: torch.Tensor
            A tensor of shape ``[batch_size, seq_len, state_dim]``.
        actions: torch.Tensor (optional)
            A tensor of shape ``[batch_size, seq_len, action_dim]``. If not
            provided, the actions will be computed as the change from one
            state to the next.
        
        Returns
        -------
        posterior: Normal
            A Gaussian distribution over the latent space parameterized by
            the mean and log variance (denoted above as :math:`q_{\phi}(z|x)`)
        """
        # Compute actions as change from state to state
        actions = actions if actions else states[1:] - states[:-1]
        
        # Concatenate states and actions for input to the model
        enc_birnn_input = torch.cat([states[:-1], actions], dim=-1)
        output, hiddens = self.enc_birnn(enc_birnn_input)
      
        # If using the final hidden state - forward AND backward final hidden
        if hasattr(self, "final_hidden") and self.final_hidden:
            enc_fc_input = torch.cat([hiddens[-1], hiddens[-2]], dim=1)
        else:
            enc_fc_input = torch.mean(output, dim=0)

        # Infer the mean and log variance of the posterior
        enc_h = self.enc_fc(enc_fc_input)
        enc_mean = self.enc_mean(enc_h)
        enc_logvar = self.enc_logvar(enc_h)

        posterior = Normal(enc_mean, enc_logvar)
        
        return posterior
