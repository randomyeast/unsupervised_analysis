import os
import torch
import torch.nn as nn
from glob import glob

from lib.util.log import LogEntry

class BaseSequentialModel(nn.Module):
    """Abstract model class that the other models, which are defined as
       submodules, inherit from.
    """ 
    model_args = []
    
    def __init__(self, model_config):
        """Initializes model attributes specified in the `model_config`
        
        Parameters
        ----------
        model_config: dict
            A dictionary containing the following model attributes:
                - `recurrent`: bool
                    Whether or not to make the encoder and decoder recurrent.
                - `rnn_dim`: int
                    An integer defining the number of features to use in the
                    hidden units within a recurrent encoder and decoder, if
                    using.
                - `num_layers`: int
                    The number of recurrent layers to use.
                - `label_dim`: the dimensionality of the labels used for
                    decoding tasks e.g. 3 if using 3 bins. 

        """
        # Call nn.Module initialization
        super().__init__()

        # All models should be recurrent, so always check for these
        self.model_args.append('rnn_dim') 
        self.model_args.append('num_layers')

        # Assert label_dim is defined if model requires labels
        if self.requires_labels and 'label_dim' not in self.model_args:
            self.model_args.append('label_dim')

        # Check for missing arguments
        missing_args = set(self.model_args) - set(model_config)
        assert len(missing_args) == 0, \
            'model_config is missing these arguments:\n\t{}'.format(
                ', '.join(missing_args)
            )
       
        # TODO: Is this needed? Can't we just set model attributes instead? 
        self.config = model_config

        self.log = LogEntry()

        # Some models require multi-stage training
        self.stage = 0 

        self._construct_model()
        self._define_losses()
        self._init_optimizer()

    def _construct_model(self):
        """Should be defined in submodule e.g. TVAE"""
        raise NotImplementedError

    def _define_losses(self):
        """Should be defined in submodule e.g. TVAE"""
        raise NotImplementedError

    @property
    def num_parameters(self):
        """Counts the learnable parameters in the model
        
        Returns
        -------
        self._num_parameters: int 
            If the parameters have not already been counted, count them
            store the resulting value as self._num_parameters, and return.
            
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = 0 
            for p in self.parameters():
                count = 1 
                for s in p.size():
                    count *= s
                self._num_parameters += count

        return self._num_parameters

    def init_hidden_state(self, batch_size=1):
        """Initializes the hidden state to zeros for a recurrent neural network
        
        Parameters
        ----------
        batch_size: int, optional
            TODO: Used to set one of the dimensions of the recurrent unit -
            EXPLAIN WHAT THIS MEANS!

        """
        return torch.zeros(self.num_layers, batch_size, self.rnn_dim)

    def update_hidden(self, state, action):
        """Single step update of the hidden unit in the reconstruction decoder
        
        Parameters
        ----------
        state: torch.tensor
            current state being passed as input to the recurrent unit
        action: torch.tensor
            the change from the current state which will produce the 
            following state.

        Returns
        -------
        hiddens: torch.tensor
            TODO: the output of the final layer in an RNN? Maybe add some 
            sort of diagram explaining this?
        """
        assert self.is_recurrent
        state_action_pair = torch.cat([state, action], dim=1).unsqueeze(0)
        hiddens, self.hidden = self.dec_rnn(state_action_pair, self.hidden)

        return hiddens

    def _init_optimizer(self, lr=1e-4):
        """Initializes the optimizer for the model"""
        self.model_optimizer = torch.optim.Adam(self.model_params(), lr=lr)

    def optimize(self, losses, grad_clip=10):
        """
        Optimizes the model parameters using the provided losses

        Parameters
        ----------
        losses: dict
            A dictionary of losses to be optimized. Should be located in the
            `self.log.losses` attribute of whatever model you are working with.
        grad_clip: float, optional
            The maximum gradient magnitude to clip to. RNNs sometimes have issues
            with exploding gradients, this is a way to prevent them.

        """
        self.model_optimizer.zero_grad()
        model_losses = [value for key, value in losses.items()]
        model_loss = sum(model_losses)
        model_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model_params(), grad_clip)
        self.model_optimizer.step()

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

    def prepare_stage(self, config_dir):
        """
        Loads the best checkpoint from the previous stage, when moving to
        the next stage. Also creates directory for storing checkpoints in
        the next stage.
       
        Parameters
        ----------
        config_dir: str
            The path to the directory for this experiment.
        """
        if self.stage > 0:
            self.load_best_checkpoint(config_dir)
        stage_path = os.path.join(config_dir, 'checkpoints', f'stage_{self.stage}')
        os.mkdir(stage_path)