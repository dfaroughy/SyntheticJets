import numpy as np
from datamodule_jetclass import JetSequence
from utils import binnify

jets = JetSequence(max_seq_length=40)

tokens = np.load('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_epicfm/gen_TTBar_seq_epicfm_tokens.npy')[:, :42]

digits = jets.seq_to_bins_decoding(tokens)
bins = binnify(digits, make_continuous=True)
np.save('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_epicfm/gen_TTBar_seq_epicfm_binned_smeared.npy', bins.numpy())
print(bins[0])
print(bins[10])
print(bins[100])


# load h5 file

# import h5py
# with h5py.File('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/TTBar_EPiC_FM_bins403030.h5', 'r') as f:
#     arr = f['discretized/block0_values']
#     digits = arr[:]
#     digits = digits.reshape(digits.shape[0], -1, 3)

# bins = binnify(digits, make_continuous=True)
# np.save('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/TTBar_EPiC_FM_bins403030_smeared.npy', bins.numpy())
# print(bins[0])
# print(bins[10])
# print(bins[100])