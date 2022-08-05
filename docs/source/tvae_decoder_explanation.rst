TVAE decoder explanation
========================
.. tvae_decoder_explanation:

.. _recurrent-decoder:
.. image:: images/decoder.png
    :align: center

**Recurrent portion of decoder**

*   The recurrent portion of the decoder is similar to the
    recurrent portion of the encoder but instead of using
    a state and an action as inputs at each timestep (shown
    :doc:`here <tvae_encoder_explanation>`), it uses a state, latent
    variable z, and the hidden output from the previous timestep.  

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
