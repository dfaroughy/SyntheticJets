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
        shape_param=1.0,   # k hyper-param in Gamma(beta|k) prior
        scale_param=1.0,   # Sigma hyper-param in N(phi|0,Sigma) prior
        tokenize=False,    # True: tokenized constituents
        z_order=False,     # True: canonically ordered
        bins_z=None,       # bins for tokenization
        bins_phi=None,     # bins for tokenization
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
            beta = np.random.gamma(shape=self.shape)   
            # mean = np.random.normal(loc=0.0, scale=self.scale)

            event_points = np.zeros((num_constituents, 2))

            for j in range(num_constituents):
                z = np.random.beta(0.5, 1 + beta)
                phi = np.random.normal(loc=self.scale * z - beta, scale=z)
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

    def log_prob(self, sample, alpha=0.5):
        """
        sample: shape (N, D, 2)
        returns array of length N of log‐prob per jet.
        """
        Sigma = self.scale
        k = self.shape
        return np.array([self._log_likelihood(sample[i], k, Sigma, alpha)
                        for i in range(len(sample))])

    def tokens_to_bins(self, tokenized_sample):
        # Compute the individual bin indices.
        z_bin = tokenized_sample % self.num_z_bins
        phi_bin = tokenized_sample // self.num_z_bins
        z_bin = np.clip(z_bin, 0, len(self.bins_z) - 2)
        phi_bin = np.clip(phi_bin, 0, len(self.bins_phi) - 2)
        z_center = (self.bins_z[z_bin] + self.bins_z[z_bin + 1]) / 2
        phi_center = (self.bins_phi[phi_bin] + self.bins_phi[phi_bin + 1]) / 2
        return np.stack((z_center, phi_center), axis=-1)


    def log_probs(self, sample, alpha=0.5):
            """
            sample: shape (N, D, 2)
            returns array of length N of log‐prob per jet.
            """
            k = self.shape
            scale = self.scale
            return np.array([
                self._log_likelihood(sample[i], k, scale, alpha)
                for i in range(len(sample))
            ])

    def _log_likelihood(self, sample, k, scale, alpha=0.5):
        """
        Jet log‐likelihood with z,phi and latent beta integrated out.
        sample:      array of shape (D,2), columns = [z, phi]
        k:           Gamma prior shape on beta
        scale:       the 'scale' hyper‑parameter
        alpha:       Beta prior first shape
        """
        z = sample[:, 0]
        phi = sample[:, 1]
        D = len(z)

        sum_log_z   = np.sum(np.log(z))
        sum_log_1mz = np.sum(np.log(1 - z))
        log_gamma_k = gammaln(k)

        def integrand(beta):
            log_p_beta = (k - 1) * np.log(beta) - beta - log_gamma_k
            A = gammaln(alpha + 1 + beta) - gammaln(alpha) - gammaln(1 + beta)
            log_p_z = D * A + (alpha - 1) * sum_log_z + beta * sum_log_1mz
            quad_term = -0.5 * (D * np.log(2 * np.pi) + 2 * sum_log_z)
            sq_err   = np.sum(((phi - (scale * z - beta))**2) / (z**2))
            log_p_phi = quad_term - 0.5 * sq_err

            return np.exp(log_p_beta + log_p_z + log_p_phi)

        val, err = quad(integrand, 0, np.inf, limit=200)

        return np.log(val)


    def _log_likelihood(self, sample, k, scale, alpha=0.5, gam=1.0):
        """
        Jet log‐likelihood with z,phi and latent beta integrated out.
        sample:      array of shape (D,2), columns = [z, phi]
        k:           Gamma prior shape on beta
        scale:       the 'scale' hyper‑parameter
        alpha:       Beta prior first shape
        """
        
        z = sample[:, 0]
        phi = sample[:, 1]
        
        D = len(z)
        x = (phi - gam * z) / z
        sum_log_z   = np.sum(np.log(z))
        sum_log_1mz = np.sum(np.log(1 - z))
        A_tilde = np.sum(x**2) - (alpha - 2) * sum_log_z 

        logp = A_tilde - 0.5 * D * np.log(2 * np.pi) - D * gammaln(alpha) - gammaln(k)

        a = 0.5 + np.sum(x) - sum_log_1mz
        b = np.sum(1/(x**2))

        def integrand(beta):
            I = (k-1) * np.log(beta) + D * gammaln(1 + alpha + beta) - D * gammaln(1 + beta)
            J =  + a * beta + b * beta**2 
            return np.exp(I + J)

        val, err = quad(integrand, 0, np.inf, limit=200)
        logp += np.log(val)
        return logp


