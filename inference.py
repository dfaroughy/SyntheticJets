import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as L

from torch.utils.data import DataLoader
from synthetic_data import SyntheticJets
from utils import kin_plots,  ordered_z_plots, ROC
from models import GPT2Model

###############################################################################
N = 100000
tag = 'fine'
signal_id = "44435059d9624828a240bc94a32f7214"
background_id = "e3e5072962ba4dde81b96c7090da422e"
path = "/pscratch/sd/d/dfarough/synthetic-jets/"
###############################################################################

#...Signal

sig_gpt2 = GPT2Model.load_from_checkpoint(f"{path}/{signal_id}/checkpoints/best.ckpt")
prompts = torch.full((N, 1), sig_gpt2.start_token, dtype=torch.long, device=sig_gpt2.device)
prompt_dataloadeer = DataLoader(prompts, batch_size=1024, shuffle=False)

sig_gpt2.top_k = 1000

generator = L.Trainer(accelerator="gpu", devices=[0])
sig_gen_seq = generator.predict(sig_gpt2, dataloaders=prompt_dataloadeer)
sig_gen_seq = torch.cat(sig_gen_seq, dim=0)
sig_gen_bin = sig_gpt2.synthetic_jets.tokens_to_bins(sig_gen_seq) 
np.save(f'{path}{signal_id}/gen_data_gpt2.npy', sig_gen_bin)

data = SyntheticJets(shape_param=sig_gpt2.shape, 
                    scale_param=sig_gpt2.scale, 
                    bins_z= sig_gpt2.bins_z,
                    bins_phi=sig_gpt2.bins_phi,
                    z_order=True,
                    tokenize=True,
                    )

jet_seq  = data.sample(N=N)
jet_bin = data.tokens_to_bins(jet_seq)
kin_plots(jet_bin, sig_gen_bin, f"{path}{signal_id}/kinematics.png")
ordered_z_plots(jet_bin, sig_gen_bin, f"{path}{signal_id}/ordered_z.png")

#############################################

#...Background

bkg_gpt2 = GPT2Model.load_from_checkpoint(f"{path}/{background_id}/checkpoints/best.ckpt")
prompts = torch.full((N, 1), bkg_gpt2.start_token, dtype=torch.long, device=bkg_gpt2.device)
prompt_dataloadeer = DataLoader(prompts, batch_size=1024, shuffle=False)

bkg_gpt2.top_k = 1000

generator = L.Trainer(accelerator="gpu", devices=[0])
bkg_gen_seq = generator.predict(bkg_gpt2, dataloaders=prompt_dataloadeer)
bkg_gen_seq = torch.cat(bkg_gen_seq, dim=0)
bkg_gen_bin = bkg_gpt2.synthetic_jets.tokens_to_bins(bkg_gen_seq) 
np.save(f'{path}{background_id}/gen_data_gpt2.npy', bkg_gen_bin)

data = SyntheticJets(shape_param=bkg_gpt2.shape, 
                    scale_param=bkg_gpt2.scale, 
                    bins_z= bkg_gpt2.bins_z,
                    bins_phi=bkg_gpt2.bins_phi,
                    z_order=True,
                    tokenize=True,
                    )

jet_seq  = data.sample(N=N)
jet_bin = data.tokens_to_bins(jet_seq)
kin_plots(jet_bin, bkg_gen_bin, f"{path}{background_id}/kinematics.png")
ordered_z_plots(jet_bin, bkg_gen_bin, f"{path}{background_id}/ordered_z.png")

#############################################

bkg_logp_on_qcd = bkg_gpt2.log_probs(bkg_gen_seq, batch_size=1024)
bkg_logp_on_tops = bkg_gpt2.log_probs(sig_gen_seq, batch_size=1024)
sig_logp_on_qcd = sig_gpt2.log_probs(bkg_gen_seq, batch_size=1024)
sig_logp_on_tops = sig_gpt2.log_probs(sig_gen_seq, batch_size=1024)

LLR_qcd = bkg_logp_on_tops - sig_logp_on_tops     
LLR_tops = bkg_logp_on_qcd - sig_logp_on_qcd  

optimal_LLR_qcd = np.nan_to_num(np.load(f"{path}/data/LLR_data_bkg.npy"), nan=0.0)
optimal_LLR_tops = np.nan_to_num(np.load(f"{path}/data/LLR_data_sig.npy"), nan=0.0)

plt.figure(figsize=(3, 3))
ROC(optimal_LLR_qcd, optimal_LLR_tops, "optimal data")
ROC(LLR_qcd, LLR_tops, "optimal GPT2")
plt.legend(fontsize=6, loc="lower left")
plt.savefig(f'ROC_{tag}.png', dpi=300, bbox_inches='tight')
