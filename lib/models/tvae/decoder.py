import torch
import torch.nn as nn

from lib.distributions import Normal

class TVAEDecoder(nn.Module):

    # TODO: Define the default model parameters
    
    def __init__(self, log, **kwargs):
        """
        Define the decoder network architecture

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
        """
        TODO: make the input shape arguments follow the PyTorch convention

        states should be in shape [seq_len, batch_size, state_dim*2]

        """
        # TODO: add method for resetting policy, this will change z
        self.reset_policy(z) 

        # Compute actions
        actions = states[1:] - states[:-1]

        # Iterate through state reconstruction
        curr_state, curr_action = states[0], actions[0]
        for t in range(actions.size(0)-1):
            # Compute distribution of actions
            action_likelihood = self.decode_action(curr_state)

            # Compute the nll of the true action under the predicted distribution
            self.log.losses['nll'] -= action_likelihood.log_prob(curr_action)

            # Update hidden state in recurrent portion of decoder
            self.update_hidden_state(curr_state, curr_action)
            if hasattr(self, "teacher_force") and self.teacher_force:
                curr_state, curr_action = states[t+1], actions[t+1]
            # Use rollout procedure to succesively synthesize states
            else:
                curr_action = action_likelihood.sample()
                curr_state = curr_state + curr_action


    def decode_action(self, state):
        # Inputs to recurrent unit: state, embedding, hidden state
        dec_fc_input = torch.cat([state, self.z, self.hidden[-1]], dim=1)
        
        # Compute hidden state from recurrent state
        dec_h = self.dec_action_fc(dec_fc_input)

        # Compute Gaussian distribution of actions
        dec_mean = self.dec_action_mean(dec_h)
        dec_logvar = self.dec_action_logvar(dec_h)

        return Normal(dec_mean, dec_logvar)

    def reset_policy(self, z, temperature=1.0):
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