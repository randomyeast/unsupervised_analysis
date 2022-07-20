import torch
import torch.nn as nn

from lib.distributions import Normal

class TVAEEncoder(nn.Module):
    r"""Encoder module for the TVAE. 
        
        Notes
        -----
        The encoder consists of a bidirectional gated recurrent neural 
        network (GRU) and 3 fully connected modules. In the default model 
        configuration, the output of the final layer of the GRU is averaged
        over the sequence. This average over time is then passed through the 
        first module of fully-connected layers, followed by two separate
        fully-connected modules, one for the mean and one for the log 
        variance of the inferred posterior distribution. Calling the module
        on a batch of state sequences will return the inferred posterior 
        distribution over the latent space. More details can be found in
        the initialization method and forward method.
    
    """
    # TODO: Define the default model parameters
    
    def __init__(self, log, **kwargs):
        r"""Defines the encoder network architecture

        Notes
        =====
            **Recurrent portion of the encoder**

            To clarify available variations of the encoder, we will go over 
            PyTorch's RNN interface. GRUs are used by default in the model
            definition, but for simplicty the details of how GRUs work is
            omitted. The recurrent portion of the encoder can be thought 
            of as a recurrent neural network (RNN) described by the equations:

            .. math:: 
                \begin{center}

                    a^{(t)} = (Wh^{(t-1)} + b_{W}) + (Ux^{(t)} + b_{U})
                    \\
                    h^{(t)} = \sigma(a^{(t)})
        
                \end{center}

            where :math:`h^{(t)}` is the hidden state at time :math:`t`. The 
            matrix :math:`U` is a matrix of weights applied to the input at
            each time step, and :math:`W` is a matrix of weights applied to the
            hidden state propagated from the previous time step. :math:`b_{W}` 
            and :math:`b_{U}` are bias vectors. :math:`\sigma` is the activation
            function which, in PyTorch, defaults to hyperbolic tangent. 

            When calling ``self.enc_birnn(x)``, x should be a tensor of shape
            ``[seq_len, batch_size, state_dim*2]``. The output of 
            ``self.enc_birnn`` is a tuple of tensors ``outputs, hiddens``. The
            ``outputs`` tensor will be of shape ``[seq_len, batch_size, 
            rnn_dim]``. Indexing along the first dimension gives the value of
            :math:`h^{(t)}` for each time step. The ``hiddens`` tensor will be
            of shape ``[num_layers, batch_size, rnn_dim]``. Indexing along the 
            ``num_layers`` dimension gives the computed hidden state at the 
            final time step for each layer in the RNN.

            The difference between the two model variations available for use
            is what output from the recurrent portion of the encoder is used 
            as an input to the fully-connected portion of the encoder. *(1)* 
            uses an average :math:`h^{(t)}` over each time step and *(2)* uses
            the output from the final hidden layer at the final time step.

        Parameters
        ========== 
        kwargs: dict
            A dictionary containing the following model attributes:
                - `state_dim`: int
                    The dimensionality of the input space i.e. the number of 
                    elements in each input state vector, and the number of 
                    columns in :math:`U`.
                - `rnn_dim`: int
                    The dimensionality of the hidden state of the GRU. This
                    corresponds to the number of rows in :math:`U` and the 
                    dimensionality of the hidden states.
                - `num_layers`: int
                    The number of layers to use in the recurrent portion of 
                    the encoder.
                - `h_dim`: int
                    The dimensionality of the space mapped to by the first
                    two fully-connected layers in the encoder ``self.enc_fc``
                - `z_dim`: int
                    The dimensionality of the latent space mapped to by the
                    separate fully-connected layers in the encoder
                    ``self.enc_mu`` and ``self.enc_logvar``.

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
            A tensor of shape ``[seq_len, batch_size, state_dim]``.
        actions: torch.Tensor (optional)
            A tensor of shape ``[seq_len, batch_size, action_dim]``. If not
            provided, the actions will be computed as the change from one
            state to the next.
        
        Returns
        -------
        posterior: Normal
            A Gaussian distribution over the latent space parameterized by
            the mean and log variance.
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
