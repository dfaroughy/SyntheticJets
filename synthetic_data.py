import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import gammaln, gamma
from torch.utils.data import Dataset


class JetSequenceDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_constituents: int = 30,
        shape_param: float = 1.0,
        scale_param: float = 1.0,
        bins_z: np.ndarray = np.linspace(0, 1, 10),
        bins_phi: np.ndarray = np.linspace(-4, 4, 10),
    ):
        """
        A PyTorch Dataset that produces synthetic jet sequences on the fly.

        Parameters:
          num_samples (int): Total number of jet events in the dataset.
          num_constituents (int): Number of jet constituents per event (excluding start token).
          shape_param (float): Shape parameter for the Gamma distribution.
          scale_param (float): Scale parameter for the Normal distribution.
          z_order (bool): Whether to sort constituents by descending z before tokenizing.
          bins_z (np.ndarray): Bin edges for z.
          bins_phi (np.ndarray): Bin edges for phi.
        """

        self.num_samples = num_samples
        self.num_constituents = num_constituents
        self.bins_z = bins_z
        self.bins_phi = bins_phi
        self.synthetic_jets = SyntheticJets(
            shape_param=shape_param,
            scale_param=scale_param,
            tokenize=True,
            z_order=True,
            bins_z=bins_z,
            bins_phi=bins_phi,
        )

        num_z_bins = len(bins_z) - 1
        num_phi_bins = len(bins_phi) - 1
        self.vocab_size = num_z_bins * num_phi_bins
        self.start_token = self.vocab_size  # reserved start token index

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.synthetic_jets.sample(N=1, num_constituents=self.num_constituents)
        seq = seq.squeeze(0)  # shape (num_constituents,)
        seq = np.concatenate([[self.start_token], seq])  # add start tokens
        return torch.tensor(seq, dtype=torch.long)


class SyntheticJets:
    def __init__(
        self,
        shape_param=1.0,
        scale_param=1.0,
        tokenize=False,
        z_order=False,
        bins_z=None,
        bins_phi=None,
    ):
        self.shape = shape_param
        self.scale = scale_param
        self.tokenize = tokenize
        self.z_order = z_order

        if tokenize:
            self.bins_z = bins_z
            self.bins_phi = bins_phi
            self.num_z_bins = len(self.bins_z) - 1

    def sample(self, N, num_constituents=30):
        samples = (
            np.zeros((N, num_constituents, 2))
            if not self.tokenize
            else np.zeros((N, num_constituents), dtype=int)
        )

        for i in range(N):
            beta = np.random.gamma(
                shape=self.shape
            )  # np.random.gamma uses scale = 1/rate
            mean = np.random.normal(loc=0.0, scale=self.scale)

            event_points = np.zeros((num_constituents, 2))

            for j in range(num_constituents):
                z = np.random.beta(0.5, 1 + beta)
                phi = np.random.normal(loc=mean, scale=1.0)
                event_points[j, :] = [z, phi]

            if self.z_order:
                event_points = event_points[np.argsort(event_points[:, 0])][::-1]

            if self.tokenize:
                for j in range(num_constituents):
                    z, phi = event_points[j, :]
                    z_bin = np.digitize(z, self.bins_z) - 1
                    phi_bin = np.digitize(phi, self.bins_phi) - 1
                    z_bin = np.clip(z_bin, 0, len(self.bins_z) - 2)
                    phi_bin = np.clip(phi_bin, 0, len(self.bins_phi) - 2)
                    token = phi_bin * self.num_z_bins + z_bin
                    samples[i, j] = token
            else:
                samples[i, :, 0] = event_points[:, 0]
                samples[i, :, 1] = event_points[:, 1]

        return samples

    def log_prob(self, sample):
        z = sample[:, 0]
        phi = sample[:, 1]
        return self._log_p_z(z, self.shape) + self._log_p_phi(phi, self.scale)

    def tokens_to_bins(self, tokenized_sample):
        # Compute the individual bin indices.
        z_bin = tokenized_sample % self.num_z_bins
        phi_bin = tokenized_sample // self.num_z_bins
        z_bin = np.clip(z_bin, 0, len(self.bins_z) - 2)
        phi_bin = np.clip(phi_bin, 0, len(self.bins_phi) - 2)
        z_center = (self.bins_z[z_bin] + self.bins_z[z_bin + 1]) / 2
        phi_center = (self.bins_phi[phi_bin] + self.bins_phi[phi_bin + 1]) / 2
        return np.stack((z_center, phi_center), axis=-1)

    def _log_p_phi(self, x, scale):
        # Computes log p(y) from the marginal over the latent mean.
        # Assume y_i ~ N(mu_lat,1) with mu_lat ~ N(0, sigma^2), mu=0.
        D = len(x)
        sum_x = np.sum(x)
        sum_x2 = np.sum(x**2)
        term1 = D * np.log(2 * np.pi)
        term2 = np.log(1 + D * scale**2)
        term3 = sum_x2 - (sum_x**2) / (D + 1 / scale**2)
        return -0.5 * (term1 + term2 + term3)

    def _log_p_z(self, x, shape):
        """
        Computes the log marginal likelihood for the x-data,
        where for a given sample of D x-values:

        x_i ~ Beta(0.5, 1+alpha)   with likelihood:
            p(x|alpha) = (Gamma(alpha+beta)/(Gamma(0.5)*Gamma(1+alpha))) * x^(-0.5)*(1-x)^(alpha)

        and
        alpha ~ Gamma(k,1)   with density:  p(alpha)=alpha^(k-1)*exp(-alpha)/Gamma(k)

        We define:
            S = sum_i ln(1-x_i)
        and there is an extra constant from the x^(-0.5) factors.
        """
        D = len(x)
        T = np.sum(np.log(x))
        S = np.sum(np.log(1 - x))

        A = -0.5 * T - np.log(gamma(shape)) - D * 0.5 * np.log(np.pi)

        def integrand(beta):
            log_ratio = gammaln(1.5 + beta) - gammaln(1 + beta)
            return np.exp(-beta * (1 - S)) * np.exp(D * log_ratio) * beta ** (shape - 1)

        val, _ = quad(integrand, 0, np.inf, limit=100)
        return A + np.log(val)

