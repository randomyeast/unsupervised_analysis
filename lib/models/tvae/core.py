import os
import torch
from glob import glob
from tqdm import tqdm

from lib.distributions import Normal
from lib.util.log import LogEntry
from lib.models import BaseSequentialModel
from lib.models.tvae.encoder import TVAEEncoder
from lib.models.tvae.decoder import TVAEDecoder

class TVAE(BaseSequentialModel):
    # Set default loss weights
    loss_params = {
        "nll": 1.0,
        "kld": 1.0
    }

    # TVAE does not require labels
    requires_labels = False

    def __init__(self, model_config):
        """
        Trajectory Variational Autoencoder (TVAE): relatively simple 
        variational autoencoder which learns lower dimensional embeddings
        of fixed length trajectories.

        Attributes
        ==========
        - encoder : TVAEEncoder
            Infers a posterior distribution over the latent space
        - decoder : TVAEDecoder
            Reconstructs the input states using a sample from the inferred posterior


        Parameters
        ========== 
        model_config: dict
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

        super().__init__(model_config)

    def _construct_model(self):
        """
        Creates encoder and decoder model attributes. Called by the 
        ``__init__`` method in ``BaseSequentialModel``.
        """

        # Override defaults using model config, if desired
        for param_key in self.loss_params.keys():
            if param_key in self.config.keys():
                self.loss_params[param_key] = self.config[param_key]

        # Define TVAE architecture
        self.encoder = TVAEEncoder(self.log, **self.config)
        self.decoder = TVAEDecoder(self.log, **self.config)

        # Initialize the model's training stages
        self.stage = 0

    def _define_losses(self):
        """
        Creates loss entry in ``self.log`` object. Called by the ``__init__`` 
        method in ``BaseSequentialModel``.
        """
        self.log.add_loss("nll")
        self.log.add_loss("kld")

    def forward(self, states, embed=False):
        """Initialize the distribution with mean and logvar

        Parameters
        ----------
        reconstruct : bool 
            If true, return the reconstructed states

        embed: bool
            If true, return the mean of the inferred posterior
        """
        # Reset the loss - code block OK
        self.log.reset()
        states = states.transpose(0,1)

        # Encode the states
        posterior = self.encoder(states)
        
        # Compute kl divergence from unit Gaussian prior
        kld = Normal.kl_divergence(posterior, free_bits=1/self.config['z_dim'])
        self.log.losses["kld"] = torch.sum(kld)

        # Use sample from posterior to generate a reconstruction
        if self.stage == 0:
            self.decoder(states, posterior.sample())      
        else:
            nll, _ = self.decoder.generate_rollout(states, posterior.sample())
            self.log.losses["nll"] = nll

        return self.log 

    def reconstruct(self, data_loader, num_samples=10, device=torch.device('cpu')):
        """Reconstructs the input states using samples from the posterior
        
        Parameters
        ----------
        states : torch.tensor
            Tensor of shape [num_trajs, traj_len, num_feats]  
        num_samples : int
            The number of samples to draw from the posterior to reconstruct     
            the input. The reconstruction with the lowest NLL will be returned.

        """
        with torch.no_grad():
            recons = []
            for batch_idx, states in enumerate(tqdm(data_loader)):
                # Transpose the states to shape expected by generate_rollout
                states = states.transpose(0,1).to(device)

                # Encode the states
                posterior = self.encoder(states)

                # Use sample from posterior to rollout samples and pick the best one
                best_nll = torch.inf 
                for _ in range(num_samples): 
                    nll, reconstruction = self.decoder.generate_rollout(
                        states,
                        posterior.sample(), 
                    )
                    if nll < best_nll:
                        best_recon = reconstruction
                        best_nll = nll

                recons.append(best_recon.transpose(0,1))

            return torch.cat(recons, dim=0) 

    def embed(self, data_loader, device=torch.device('cpu')):
        """Reconstructs the input states using samples from the posterior
        
        Parameters
        ----------
        states : torch.tensor
            Tensor of shape [num_trajs, traj_len, num_feats]  

        """
        zs = []
        with torch.no_grad():
            for batch_idx, states in enumerate(tqdm(data_loader)):
                # Transpose the states to shape expected by generate_rollout
                states = states.transpose(0,1).to(device)

                # Encode the states
                posterior = self.encoder(states)

                zs.append(posterior.mean)

            return torch.cat(zs, dim=0)                


    def fit(self, data_loader, device=torch.device('cuda')):
        """Fits the model using the provided data
        
        Parameters
        ---------- 
        data_loader : `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
            A wrapper around a TrajectoryDataset object which allows for easy random sampling and batching.
        
        device (optional) : `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
            Device to perform model computations on.
        """
        data_loader.dataset.train()
        self.train()
        epoch_log = LogEntry()
        for batch_idx, states in enumerate(tqdm(data_loader)):
            batch_log = self(states.to(device))
            self.optimize(batch_log.losses)
            batch_log.itemize()
            epoch_log.absorb(batch_log)

        epoch_log.average(N=len(data_loader.dataset))

        return epoch_log
 
    def test(self, data_loader, device=torch.device('cuda')):
        """Evaluates the model using the provided data
        
        Parameters
        ---------- 
        data_loader : `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
            A wrapper around a TrajectoryDataset object which allows for easy random sampling and batching.
        
        device (optional) : `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
            Device to perform model computations on.
        """
        data_loader.dataset.eval()
        self.eval() 
        with torch.no_grad():
            epoch_log = LogEntry()
            for batch_idx, states in enumerate(tqdm(data_loader)):
                batch_log = self(states.to(device))
                batch_log.itemize()
                epoch_log.absorb(batch_log)

            epoch_log.average(N=len(data_loader.dataset))

        return epoch_log

    def save_checkpoint(self, config, name='model'):
        """Saves a model checkpoint to the experiment directory, using the correct naming convention

        Parameters
        ---------- 
        config : str 
            The path to the directory for this experiment.
        name : str
            The name to save the checkpoint to.

        """

        if 'best' in name:
            name = name.split('_')[0]
            path = os.path.join(os.getcwd(), config, 'checkpoints', f'stage_{self.stage}', f'{name}.pt')
        else:
            path = os.path.join(os.getcwd(), config, 'checkpoints', f'stage_{self.stage}', f'{name}.pt')
        torch.save(self.state_dict(), path)

    def load_best_checkpoint(self, config_dir):
        """Loads a model checkpoint from the experiment directory, using the correct naming convention

        Parameters
        ---------- 
        config : str 
            The path to the directory for this experiment.
        name : str
            The name to load the checkpoint from.
        """
        stage_dirs = [d for d in glob(f'{config_dir}/checkpoints/*')]
        last_stage = stage_dirs[-1]
        best_path = os.path.join(last_stage, 'best.pt')
        self.load_state_dict(torch.load(best_path))


    def model_params(self):
        """
        Returns a list of all model parameters - used to optimize the learnable parameters"""
        params = list(self.encoder.enc_birnn.parameters()) + \
                list(self.encoder.enc_fc.parameters()) + \
                list(self.encoder.enc_mean.parameters()) + \
                list(self.encoder.enc_logvar.parameters()) + \
                list(self.decoder.dec_birnn.parameters()) + \
                list(self.decoder.dec_action_fc.parameters()) + \
                list(self.decoder.dec_action_mean.parameters()) + \
                list(self.decoder.dec_action_logvar.parameters())

        return params
