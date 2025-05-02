import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import gammaln, gamma
from torch.utils.data import Dataset
import h5py
import pandas as pd


class JetSequenceDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        num_jets: int = None,
        max_seq_length: int = 200,  # dataset max num consituents
        bins: list = [41, 31, 31],
    ):
        self.filepath = filepath
        self.num_jets = num_jets
        self.num_features = len(bins) 
        self.bins = bins
        self.num_bins = int(np.prod(self.bins))

        # special tokens:

        self.start_token = self.num_bins + 1  
        self.end_token = self.num_bins + 2       
        self.pad_token = self.num_bins + 3   

        jet_sequence = JetSequence(filepath=filepath,
                                   num_jets=num_jets,
                                   max_seq_length=max_seq_length,
                                   bins=bins,
                                   start_token=self.start_token,
                                   end_token=self.end_token,
                                   pad_token=self.pad_token)    

        seq = jet_sequence.map_to_sequence()  # shape (N, D)
        self.input_ids = torch.from_numpy(seq).long()
        self.attention_mask = (self.input_ids != self.pad_token).long()
        self.multiplicity = self.attention_mask.sum(axis=1)
        self.end_token_idx = self.multiplicity

        for i in range(self.input_ids.size(0)):
            self.input_ids[i][self.multiplicity[i]] = self.end_token
            self.attention_mask[i][self.multiplicity[i]] = 1

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],      
            "attention_mask": self.attention_mask[idx],
        }



class JetSequence:
    def __init__(
        self,
        filepath: str,
        num_jets: int = None,
        max_seq_length: int = 200,  # dataset max num consituents
        bins: list = [41, 31, 31],
        start_token: int = None,
        end_token: int = None,
        pad_token: int = -1,
    ):
        self.filepath = filepath
        self.num_jets = num_jets
        self.num_features = len(bins) 
        self.bins = bins
        self.max_seq_length = max_seq_length   

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        print(f"INFO: start token: {self.start_token}")
        print(f"INFO: end token: {self.end_token}")
        print(f"INFO: pad token: {self.pad_token}")

                    
    def map_to_sequence(self):

        with h5py.File(self.filepath, "r") as f:
            arr = f['discretized/block0_values']
            data = arr[:] if self.num_jets is None else arr[: self.num_jets]
                
        df = pd.DataFrame(data)
        x = df.to_numpy() 
        x = x.reshape(x.shape[0], -1, self.num_features)
        N, D, _ = x.shape

        seq = self.bins_to_seq_encoding(x)
        if self.start_token is not None:
            start = np.full((N, 1), self.start_token)
            seq = np.concatenate((start, seq), axis=1) 

        seq[seq < 0] = self.pad_token

        return seq[:, :-1] # rm last dummy column to keep seq_length=200

    def bins_to_seq_encoding(self, x):
        """ encode the 3-D binned jet constituents into a 1-D sequence of tokens 
        """
        seq = (x[..., 0] * self.bins[1] + x[..., 1]) * self.bins[2] + x[..., 2]
        return seq
    
    def seq_to_bins_decoding(self, seq):
        """
        Decode a 1-D sequence of token IDs back into bin triplets ids (a,b,c).
        """
        pt, eta, phi = self.bins
        eta_phi = eta * phi
        a = seq // (eta * phi)
        rem = seq % (eta * phi)
        b = rem // phi
        c = rem % phi

        b[a < 0] = -1
        c[a < 0] = -1

        return np.stack([a, b, c], axis=-1)


class SyntheticJetSequenceDataset(Dataset):
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


