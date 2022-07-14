import math
import torch

from .core import Distribution


class Normal(Distribution):
    r"""Object representing a Gaussian distribution

    Parameters
    ----------
    mean : torch.tensor
        Vector defining the mean of the Gaussian. 
    logvar: torch.tensor
        Vector defining the log-variance of the Gaussian. 
    """

    logvar_min = -16
    logvar_max = 16

    def __init__(self, mean, logvar):
        super().__init__()

        self.mean = mean
        self.logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

    def sample(self, temperature=1.0):
        r"""Generates a Gaussian distributed random sample.
        
        Notes
        -----
        Sample is generated using the re-parameterization trick
        to make backpropagation possible. Log of the variance of
        the posterior is estimated using the encoder, then std
        is computed from that:
        
        .. math:: z = \mu + \epsilon \cdot \sigma \cdot T
        
        Parameters
        ----------
        temperature: float, optional
            Additional scaling parameter for variance of the distribution.
        """
        std = torch.exp(self.logvar / 2)
        eps = torch.randn_like(std)
        return self.mean + eps * std * temperature

    def log_prob(self, value):
        r"""Computes the negative log-likelihood of a value given the distribution. 
        
        Notes
        -----
        The negative log-likelihood (NLL) shown below, is given by the log of
        the probability density function. Note that in the actual computation 
        we use the inferred log-variance output by the encoder instead of the
        standard deviation.
 
        .. math:: \ln[\sigma] + \frac{1}{2}\ln[2\pi] + 
                \frac{(x-\mu)^{2}}{2\sigma^{2}}

        Parameters
        ----------
        value: torch.tensor
            The value we are computing the NLL of.       
 
        """
        pi = torch.FloatTensor([math.pi]).to(value.device)
        nll_element = (value - self.mean).pow(2) / \
            torch.exp(self.logvar) + self.logvar + torch.log(2 * pi)
        return -0.5 * torch.sum(nll_element)

    @staticmethod
    def kl_divergence(normal_1, normal_2=None, free_bits=0.0):
        r"""Computes the kl-divergence between two Gaussian distributions. 
        
        Notes
        -----
        Note that as in the NLL computation, instead of using standard deviation 
        we are using the inferred log-variance output by the encoder instead of
        the standard deviation. 
 
        .. math:: \frac{\sigma_1}{\sigma_{2}} + \frac{\sigma_{1}^2+(\mu_1 - 
                \mu_2)^{2}}{2\sigma^{2}_{2}} 

        Parameters
        ----------
        normal_1: Normal 
            The first Gaussian in the equation above
        normal_2: Normal, optional
            The second Gaussian in the equation above. If not included,assumed
            to be a unit Gaussian distribution.
        free_bits: float, optional
            Scalar value intended used to keep the inferred posterior from 
            collapsing into the unit Gaussian prior. If the KLD falls below 
            `free_bits` in a particular dimension, that dimension's KLD is 
            assigned to be `free_bits`.
        """

        assert isinstance(normal_1, Normal)
        mean_1, logvar_1 = normal_1.mean, normal_1.logvar

        if normal_2 is not None:
            assert isinstance(normal_2, Normal)
            mean_2, logvar_2 = normal_2.mean, normal_2.logvar

            kld_elements = 0.5 * (logvar_2 - logvar_1 +
                                  (torch.exp(logvar_1) + (mean_1 - mean_2).pow(2)) /
                                  torch.exp(logvar_2) - 1)
        else:
            kld_elements = -0.5 * \
                (1 + logvar_1 - mean_1.pow(2) - torch.exp(logvar_1))

        # Prevent posterior collapse with free bits
        if free_bits > 0.0:
            _lambda = free_bits * \
                torch.ones(kld_elements.size()).to(kld_elements.device)
            kld_elements = torch.max(kld_elements, _lambda)

        kld = torch.sum(kld_elements, dim=-1)

        return kld
