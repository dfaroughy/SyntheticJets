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
    ):
        self.filepath = filepath
        self.num_jets = num_jets
        self.bins = bins
        self.max_seq_length = max_seq_length

        # special tokens
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        if filepath is not None:
            # load raw binned data: shape (N, D, 3)
            self.data = self._load_data()[:, :max_seq_length, :]

    def _load_data(self):
        with h5py.File(self.filepath, "r") as f:
            arr = f['discretized/block0_values']
            raw = arr[:] if self.num_jets is None else arr[: self.num_jets]
        N, D3 = raw.shape[0], raw.shape[1]
        # reshape to (N, D, 3)
        return raw.reshape(N, -1, len(self.bins))

    def bins_to_seq_encoding(self, x: np.ndarray) -> np.ndarray:
        # combine 3D bins into flat token IDs
        a = x[..., 0]
        b = x[..., 1]
        c = x[..., 2]
        return (a * self.bins[1] + b) * self.bins[2] + c

    def map_to_sequence(self) -> np.ndarray:
        """
        Returns:
          seqs: np.ndarray of shape (N, max_seq_length + 2)
        where each row is [BOS, token_1 .. token_L, EOS, PAD, ...]
        """
        # encode raw bins
        seq = self.bins_to_seq_encoding(self.data)  # (N, D)
        N, D = seq.shape
        S = self.max_seq_length

        # initialize full-length array with PADs
        full = np.full((N, S + 2), self.pad_token, dtype=int)

        # BOS
        if self.start_token is not None:
            full[:, 0] = self.start_token

        # copy up to S data tokens
        length = D
        if length > S:
            length = S
        full[:, 1:1 + length] = seq[:, :length]

        # EOS right after last real token (or at position S+1 if truncated)
        full[:, 1 + length] = self.end_token

        return full


class JetSequenceDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        num_jets: int = None,
        max_seq_length: int = 200,
        bins: list = [41, 31, 31],
    ):
        # special tokens
        num_bins = int(np.prod(bins))
        start_token = num_bins    # BOS
        end_token = num_bins + 1  # EOS
        pad_token = num_bins + 2  # PAD

        # build sequences
        seq_builder = JetSequence(
            filepath=filepath,
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

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }

