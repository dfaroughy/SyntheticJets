import numpy as np
from datamodule_jetclass import JetSequence
from utils import binnify

# jets = JetSequence(max_seq_length=40)

# tokens = np.load('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_epicfm/gen_TTBar_seq_epicfm_tokens.npy')[:, :42]

# digits = jets.seq_to_bins_decoding(tokens)
# bins = binnify(digits, make_continuous=True)
# np.save('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_epicfm/gen_TTBar_seq_epicfm_binned_smeared.npy', bins.numpy())
# print(bins[0])
# print(bins[10])
# print(bins[100])

#################

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

#################

# import h5py

# with h5py.File('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/ZJetsToNuNu_EPiC_FM_bins403030.h5', 'r') as f:
#     arr = f['discretized/block0_values']
#     digits = arr[:]
#     digits = digits.reshape(digits.shape[0], -1, 3)

# vocab_size = 41 * 31 * 31

# jets = JetSequence(data=digits,
#                    start_token=vocab_size, 
#                    end_token=vocab_size + 1, 
#                    pad_token=vocab_size + 2, 
#                    max_seq_length=40)

# tks = jets.map_to_sequence()
# np.save('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/ZJetsToNuNu_EPiC_FM_tokens.npy', tks)
# print(tks[0])
# print(tks[10])
# print(tks[100])
# print(tks.shape)


#################


# logp = np.load('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/GPT2_logp/ZJetsToNuNu_logp_ZJetsToNuNu_epicfm_10k.npy')
# print(logp.shape)
# print(logp[0:10])

# # plot logp histogram

# import matplotlib.pyplot as plt
# plt.hist(logp, bins=30, density=True)
# plt.xlabel('Log Probability')
# plt.ylabel('Density')
# plt.title('Log Probability Histogram')
# plt.grid()
# plt.savefig('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/GPT2_logp/ZJetsToNuNu_logp_ZJetsToNuNu.png')


#################


data = np.load('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/ZJetsToNuNu_EPiC_FM_tokens.npy')[11_990_000:12_000_000]
print(data.shape)
print((data<39401).sum(axis=1)[:10])

