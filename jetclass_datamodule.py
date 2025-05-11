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
        self.data = self.get_data()  # shape (N, D, 3)

    def get_data(self):
        with h5py.File(self.filepath, "r") as f:
            arr = f['discretized/block0_values']
            data = arr[:] if self.num_jets is None else arr[: self.num_jets]
        print(f'INFO: shape of training data:{data.shape}') 
        return data.reshape(data.shape[0], -1, self.num_features)   

    def map_to_sequence(self):
        N, D, _ = self.data.shape
        seq = self.bins_to_seq_encoding(self.data)

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
        a = seq // (eta * phi)
        rem = seq % (eta * phi)
        b = rem // phi
        c = rem % phi

        b[a < 0] = -1
        c[a < 0] = -1

        return np.stack([a, b, c], axis=-1)

