from tkinter import W
import torch
import torch.nn as nn

from lib.distributions import Normal

class TVAEDecoder(nn.Module):

    # TODO: Define the default model parameters
    
    def __init__(self, log, **kwargs):
        r"""Defines the decoder network architecture

        .. _recurrent-decoder:
        .. image:: images/decoder.png
            :align: center

        Notes
        ===== 
            **Recurrent portion of decoder**

            *   The recurrent portion of the decoder is similar to the
                recurrent portion of the encoder but instead of using
                a state and an action as inputs at each timestep (shown
                in `recurrent-encoder`_), it uses a state and the latent
                variable z as shown in `recurrent-decoder`_.  

            **Fully connected portion of decoder**

            *   The output of the last recurrent unit at each timestep,
                the state corresponding to the current timestep, and the
                latent variable z are concatenated and fed into a fully
                connected layer ``dec_action_fc``.
                
            *   The output of ``dec_action_fc`` at each time step is fed
                into two separate fully connected layers ``dec_action_mean``
                and ``dec_action_logvar`` to generate the mean and log 
                variance of a distribution of actions, denoted above as
                :math:`\pi`.

            *   The reconstruction loss at each time step is computed as
                the negative log likelihood of the true action :math:`a_{t}`
                under the predicted distribution of actions :math:`\pi`. The
                calculation is explained in more detail in Normal 

            **Decoder variations**

            *   The default setting of the model is for ``teacher_force``
                to be ``False``. This means that the decoder will use an
                action sampled from the predicted distribution of actions
                at each timestep to *rollout* the trajectory used when
                computing the reconstruction loss. This process is shown in 
                `recurrent-decoder`_ as :math:`\tilde{s_{t}} = \tilde{s_{t-1}} + \tilde{a_{t-1}}`.

            *   If ``teacher_force`` is ``True``, the decoder will use the
                true state as the input to the recurrent unit at the next
                time step.

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


    def forward(self, states, z, reconstruct=False):
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

        # Iterate through state reconstruction
        curr_state, curr_action = states[0], actions[0]
        reconstruction = [curr_state]
        for t in range(actions.size(0)):
            # Compute distribution of actions
            action_likelihood = self.decode_action(curr_state)

            # Compute the nll of the true action under the predicted distribution
            self.log.losses['nll'] -= action_likelihood.log_prob(curr_action)

            # Update hidden state in recurrent portion of decoder
            self.update_hidden_state(curr_state, curr_action)

            # Update state and action
            if hasattr(self, "teacher_force") and self.teacher_force:
                curr_state, curr_action = states[t+1], actions[t+1]
            # Use rollout procedure to succesively synthesize states
            else:
                curr_action = action_likelihood.sample()
                curr_state = curr_state + curr_action

            # If we're reconstructing, save the current state 
            if reconstruct:
                reconstruction.append(curr_state)

        if reconstruct:
            return torch.stack(reconstruction)

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
        """
        # Set internal variables
        self.z = z
        self.temperature = temperature

        # Initialize hidden state
        self.hidden = self.init_hidden_state(
            batch_size=z.size(0)
        ).to(z.device)

    def init_hidden_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.rnn_dim)

    def update_hidden_state(self, state, action):
        state_action_pair = torch.cat([state, action], dim=1).unsqueeze(0)
        hiddens, self.hidden = self.dec_birnn(state_action_pair, self.hidden)

        return hiddens