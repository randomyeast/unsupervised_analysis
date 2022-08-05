TVAE encoder explanation
========================
.. tvae_encoder_explanation:

.. _recurrent-encoder:
.. image:: images/encoder.png
    :align: center


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

**Model variations**

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
    ``enc_mean`` and ``enc_logvar`` which learn to infer
    the mean and log variance that parameterize the posterior 
    distribution over the latent space.


