from tkinter import W
import torch
import torch.nn as nn

from lib.distributions import Normal

class TVAEDecoder(nn.Module):
    r"""Detailed explanation available :doc:`here <tvae_decoder_explanation>`"""

    # TODO: Define the default model parameters
    
    def __init__(self, log, **kwargs):
        r"""Defines the decoder network architecture

        Parameters
        ========== 
        log: lib.log.Log
            A Log object containing the losses used to train the model.
        kwargs: dict
            A dictionary containing the following model attributes:
                - `state_dim`: int
                    The dimensionality of the input space.
                - `rnn_dim`: int
                    The dimensionality of the hidden state of the GRU.
                - `num_layers`: int
                    The number of layers to use in the recurrent
                    portion of the decoder.
                - `h_dim`: int
                    The dimensionality of the space mapped to ``dec_action_fc``.
                - `z_dim`: int
                    The dimensionality of the latent space mapped to by the
                    encoder.
                - `teacher_force`: bool
                    Whether or not to use the true or rolled-out state as inputs
                    to the recurrent unit at each timestep.
        """
        super(TVAEDecoder, self).__init__()

        # Update the attributes with the passed in kwargs
        self.__dict__.update(kwargs)

        # Add log for keeping track of losses
        self.log = log

        # Define the recurrent portion of the decoder
        self.dec_birnn = nn.GRU(
            self.state_dim*2, 
            self.rnn_dim,
            num_layers=self.num_layers
        )

        # Define the fully connected portion of the decoder
        self.dec_action_fc = nn.Sequential(
            nn.Linear(self.state_dim+self.z_dim+self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

        # Define the layers predicting the distribution of actions
        self.dec_action_mean = nn.Linear(self.h_dim, self.state_dim)
        self.dec_action_logvar = nn.Linear(self.h_dim, self.state_dim)


    def forward(self, states, z):
        """Computes a reconstruction of the input states using an initial
        state and a sample from the posterior distribution. The negative
        log likelihood of the true actions under the predicted distribution
        of actions is summed over time and stored in the ``log`` attribute
        of the parent class.

        Parameters
        ----------
        states: torch.Tensor
            A tensor of shape ``[seq_len, batch_size, state_dim]`` representing
            the the same trajectory used to generate the posterior distribution
            in the encoder module.
        z: torch.Tensor
            A tensor of shape ``[batch_size, z_dim]`` representing the latent
            variable z sampled from the inferred posterior.

        """
        self.reset_policy(z) 

        # Compute actions
        actions = states[1:] - states[:-1]

        # Compute the reconstruction loss
        for t in range(actions.size(0)):
            action_likelihood = self.decode_action(states[t])
            self.log.losses['nll'] -= action_likelihood.log_prob(actions[t])
            self.update_hidden_state(states[t], actions[t])

    def generate_rollout(self, states, z):
        """
        Successively rolls out a trajectory using the previous state and an
        action sampled from a predicted distribution, given the previous state
        and the latent variable z

        Parameters
        ----------
        states : torch.Tensor
            A tensor of shape ``[trajectory_length, batch_size, state_dim]``
            representing the true states we are trying to replicate
        z: torch.Tensor
            A tensor of shape ``[batch_size, z_dim]`` representing latent variables
            z sampled from the inferred posterior distribution.

        Returns
        -------
        nll : torch.tensor
            The negative-log-likelihood computed for the rolled out trajectory
        recon: torch.tensor
            The reconstructed trajectory       
        """
        self.reset_policy(z)
        recon = [states[0]]
        actions = states[1:] - states[:-1]
        nll = 0
        for t in range(actions.size(0)):
            action_likelihood = self.decode_action(recon[-1]) 
            nll -= action_likelihood.log_prob(actions[t])
            curr_action = action_likelihood.sample()
            recon.append(recon[-1] + curr_action)
            self.update_hidden_state(recon[-1], curr_action)

        return nll, torch.stack(recon)
         
    def decode_action(self, state):
        """
        Computes the pass through the fully connected layers of the decoder

        Parameters
        ----------
        state: torch.Tensor
            A tensor of shape ``[batch_size, state_dim]`` representing the
            initial state to use in reconstructing the trajectory.
        
        Returns
        -------
        action_likelihood: Normal
            A Normal object representing the Gaussian distribution of actions
            predicted by the decoder.
        """
        # Inputs to recurrent unit: state, embedding, hidden state
        dec_fc_input = torch.cat([state, self.z, self.hidden[-1]], dim=1)
        
        # Compute hidden state from recurrent state
        dec_h = self.dec_action_fc(dec_fc_input)

        # Compute Gaussian distribution of actions
        dec_mean = self.dec_action_mean(dec_h)
        dec_logvar = self.dec_action_logvar(dec_h)

        return Normal(dec_mean, dec_logvar)

    def reset_policy(self, z, temperature=1.0):
        """ 
        Initializes the hidden state of the decoder to be all zeros
        and sets the temperature and latent variable z as attributes
        of the model

        Parameters
        ----------
        z: torch.Tensor
            A tensor of shape ``[batch_size, state_dim]`` representing the
            the sample from the inferred posterior
        """
        # Set internal variables
        self.z = z
        self.temperature = temperature

        # Initialize hidden state
        self.hidden = self.init_hidden_state(
            batch_size=z.size(0)
        ).to(z.device)

    def init_hidden_state(self, batch_size):
        """Returns a tensor of the correct shape used to initialize the RNN"""
        return torch.zeros(self.num_layers, batch_size, self.rnn_dim)

    def update_hidden_state(self, state, action):
        """Executes a forward pass through the recurrent unit at the current time step
        
        Parameters
        ----------
        state: torch.Tensor
            State from the current time step
        action: torch.Tensor
            Action used to produce the state at the current time step

        Returns
        -------
        hiddens: torch.tensor
            Output of recurrent portion of the decoder at the current time step
        """
        state_action_pair = torch.cat([state, action], dim=1).unsqueeze(0)
        hiddens, self.hidden = self.dec_birnn(state_action_pair, self.hidden)

        return hiddens
