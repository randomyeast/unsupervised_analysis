from re import I
import torch

from lib.models import BaseSequentialModel
from lib.distributions import Normal
from lib.models.tvae.encoder import TVAEEncoder
from lib.models.tvae.decoder import TVAEDecoder

class TVAE(BaseSequentialModel):
    """Trajectory Variational Autoencder
    """
    
    # Model does not use labels
    requires_labels = False
    
    # Set default loss weights
    loss_params = {
        "nll": 1.0,
        "kld": 1.0
    }

    def __init__(self, model_config):
        super().__init__(model_config)

    def _construct_model(self):
        # Override defaults using model config, if desired
        for param_key in self.loss_params.keys():
            if param_key in self.config.keys():
                self.loss_params[param_key] = self.config[param_key]

        # Define TVAE architecture
        self.encoder = TVAEEncoder(self.log, **self.config)
        self.decoder = TVAEDecoder(self.log, **self.config)
    
    def _define_losses(self):
        self.log.add_loss("nll")
        self.log.add_loss("kld")

    def forward(self, states, reconstruct=False, embed=False):
        """
        Forward pass of the model
        """
        # Encode the states
        posterior = self.encoder(states)
        
        # Compute kl divergence from unit Gaussian prior TODO: make all attrbutes available at the top-level
        kld = Normal.kl_divergence(posterior, free_bits=1/self.config['z_dim'])
        self.log.losses["kld"] = torch.sum(kld)

        # Use sample from posterior to generate a reconstruction
        self.decoder(states, posterior.sample())      

        return posterior

    def model_params(self):
        params = list(self.encoder.enc_birnn.parameters()) + list(self.encoder.enc_fc.parameters())
        params += list(self.encoder.enc_mean.parameters()) + list(self.encoder.enc_logvar.parameters())
        params += list(self.decoder.dec_birnn.parameters()) + list(self.decoder.dec_action_fc.parameters()) 
        params += list(self.decoder.dec_action_mean.parameters()) + list(self.decoder.dec_action_logvar.parameters())

        return params
