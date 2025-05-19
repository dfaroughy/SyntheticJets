import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import gammaln, gamma
from torch.utils.data import Dataset
import h5py
import pandas as pd


class JetSequence:
    def __init__(
        self,
        filepath: str = None,
        num_jets: int = None,
        max_seq_length: int = 200,  # maximum number of constituents
        bins: list = [41, 31, 31],
        start_token: int = None,
        end_token: int = None,
        pad_token: int = -1,
        get_raw: bool=False
    ):

        # special tokens
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        self.filepath = filepath
        self.num_jets = num_jets
        self.bins = bins
        self.max_seq_length = max_seq_length


        if filepath is not None:
            self.data = self._load_data()[:, :max_seq_length, :]
            self.data = self.data.astype(np.int32) # int16 is not enough for 3D bins!
            if get_raw:
                self.raw = self._load_data('raw/block0_values')[:, :max_seq_length, :]


    def _load_data(self, key='discretized/block0_values'):
        with h5py.File(self.filepath, "r") as f:
            arr = f[key]
            sample = arr[:] if self.num_jets is None else arr[: self.num_jets]
        N = sample.shape[0]
        return sample.reshape(N, -1, len(self.bins))  # (N, D, 3)

    def bins_to_seq_encoding(self, x: np.ndarray) -> np.ndarray:
        # combine 3D bins into flat token IDs
        a = x[..., 0]
        b = x[..., 1]
        c = x[..., 2]
        return (a * self.bins[1] + b) * self.bins[2] + c

    def seq_to_bins_decoding(self, seq) -> np.ndarray:
        """
        Decode a 1-D sequence of token IDs back into bin triplets ids (a,b,c).
        """
        pt, eta, phi = self.bins
        a = seq // (eta * phi)
        rem = seq % (eta * phi)
        b = rem // phi
        c = rem % phi

        b[a < 0] = -1
        c[a < 0] = -1

        return np.stack([a, b, c], axis=-1)

    def map_to_sequence(self) -> np.ndarray:
        """
        Returns:
          seqs: np.ndarray of shape (N, max_seq_length + 2)
        where each row is [BOS, token_1..token_S, EOS, PAD]
        """
        # encode raw bins
        seq = self.bins_to_seq_encoding(self.data)  # (N, S)
        N, _ = seq.shape

        start = np.full((N, 1), self.start_token, dtype=int)
        extra_pad = np.full((N, 1), self.pad_token, dtype=int)
        seq = np.concatenate([start, seq, extra_pad], axis=1)
        seq[seq < 0] = self.pad_token

        idx_eos = (seq != self.pad_token).sum(axis=1)

        for i, jet in enumerate(seq):
            jet[idx_eos[i]] = self.end_token

        return seq


class JetSequenceDataset(Dataset):
    def __init__(
        self,
        filepath: str=None,
        input_ids: str=None,
        attention_mask: str=None,
        num_jets: int = None,
        max_seq_length: int = 200,
        bins: list = [41, 31, 31],
    ):

        if filepath is not None: 
            # special tokens
            num_bins = int(np.prod(bins))
            start_token = num_bins    # BOS
            end_token = num_bins + 1  # EOS
            pad_token = num_bins + 2  # PAD

            # build sequences
            seq_builder = JetSequence(filepath=filepath,
                                    num_jets=num_jets,
                                    max_seq_length=max_seq_length,
                                    bins=bins,
                                    start_token=start_token,
                                    end_token=end_token,
                                    pad_token=pad_token,
                                    )
            seqs = seq_builder.map_to_sequence()  # (N, S+2)

            # store tensors
            self.input_ids = torch.from_numpy(seqs).long()
            self.attention_mask = (self.input_ids != pad_token).long()
        
        else:
            self.input_ids = input_ids[:, num_jets]
            self.attention_mask = attention_mask[: num_jets]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }

