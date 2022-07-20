import torch
import torch.nn as nn

from lib.distributions import Normal

class TVAEEncoder(nn.Module):
    r"""Encoder module for the TVAE"""
    # TODO: Define the default model parameters
    
    def __init__(self, log, **kwargs):
        r"""Defines the encoder network architecture

        .. _recurrent-encoder:
        .. image:: images/encoder.png
            :align: center

        Notes
        ===== 
            **Recurrent portion of encoder**

            *   While the model defaults to using Gated Recurrent Units (GRUs), 
                the recurrent portion of the encoder can be described as a network
                of simpler recurrent units, shown in `recurrent-encoder`_.
           
            *   The recurrent portion of the encoder succesively computes and 
                propagates hidden states denoted :math:`h_{t,j}` for each time
                step :math:`t` and each layer :math:`j` of the network.

            *   To give an example of how the model works, let
                :math:`x_t` be the input at time :math:`t` which is a 
                concatenation of the current state :math:`s_t` and the action
                :math:`a_t`, where :math:`a_{t}` represents the change from 
                :math:`s_t` to :math:`s_{t+1}`. To compute :math:`h_{t,0}` for
                any :math:`t` using `PyTorch's basic RNN module 
                <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>`_,
                the following equations are used. 
                
                .. math:: 
                    g_{t} = (W_{0} x_{t} + b_{W_{0}}) + (U_{0} h_{t-1} + b_{U_{0}})
               
                    h_{t} = \sigma(g_{t})

                *   :math:`W_{0}` is a matrix of learned weights mapping from 
                    the input space to the hidden space of layer 0 and
                    :math:`b_{W_{0}}` is the vector of corresponding biases.

                *   :math:`U_{0}` is a matrix of weights mapping the hidden state
                    from the previous time step to the current time step and 
                    :math:`b_{U_{0}}` is the vector of corresponding biases.
                
                *   There will be different weights :math:`W_{j}, U_{j}` and 
                    biases :math:`b_{W_{j}}, b_{U_{j}}` for each layer 

                *   :math:`\sigma` is the activation function, which when using 
                    ``torch.nn.RNN`` defaults to hyperbolic tagent.

            *   The recurrent portion of the TVAE's encoder is an attribute
                called ``enc_birnn``. When calling ``enc_birnn(x)``,x should
                be a tensor of shape ``[seq_len, batch_size,state_dim*2]``.
                The output of ``self.enc_birnn(x)`` is a tuple of tensors
                ``outputs, hiddens``.
            
            *   The ``outputs`` tensor (shown in red) will be of shape
                ``[seq_len, batch_size, rnn_dim]`` Indexing along the first
                dimension of ``outputs`` gives the value of :math:`h_{t}`
                for each time step.
                
            *   The ``hiddens`` tensor (shown above in blue) will be of shape
                ``[num_layers, batch_size, rnn_dim]``. Indexing along the
                ``num_layers`` dimension gives the computed hidden state at 
                the final time step for each layer in the RNN.

            *   There are two model variations available, each differs in what
                output of ``enc_birnn`` is passed to the fully connected
                portion of the encoder.

                *   The first variation is the default. If :math:`T` and
                    :math:`M` represent the sequence length and number of
                    layers used, respectively, this variation passes 
                    :math:`\frac{1}{T} \sum^{T} h_{t,M}` to the fully
                    connected portion of the encoder.

                *   The second variation is used when ``final_hidden`` is
                    set to ``True`` in the configuration dictionary passed to
                    the model. In this case, the hidden state at the final
                    time step and final layer :math:`h_{T,M}` is passed to the
                    fully connected portion of the encoder.

            **Fully connected portion of encoder**

            *   The output of the recurrent portion of the encoder is passed
                through two fully connected layers each with dimensionality
                specified by the ``h_dim`` parameter. Both use a ReLU
                activation function and are within an attribute called
                ``enc_fc``.

            *   The output of ``enc_fc`` is passed through two separate layers
                ``enc_mean`` and ``enc_logvar`` each of which learn to infer
                the mean and log variance that parameterize the posterior 
                distribution over the latent space.

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
                    ``enc_mu`` and ``enc_logvar``.
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
