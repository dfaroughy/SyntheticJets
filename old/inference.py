import numpy as np
import torch
import pytorch_lightning as L

from torch.utils.data import DataLoader
from synthetic_data import SyntheticJets
from utils import kin_plots,  ordered_z_plots
from models import GPT2Model

#############################################
N = 1000000
run_id = "cef5aa2802824a39b03f77dd950b8831"
#############################################

path = "/pscratch/sd/d/dfarough/tokenized-jets/"

gpt2 = GPT2Model.load_from_checkpoint(f"{path}/{run_id}/checkpoints/best.ckpt")

prompts = torch.full((N, 1), gpt2.start_token, dtype=torch.long, device=gpt2.device)
prompt_dataloadeer = DataLoader(prompts, batch_size=1024, shuffle=False)

generator = L.Trainer(accelerator="gpu", devices=[0])
jet_seq = generator.predict(gpt2, dataloaders=prompt_dataloadeer)
jet_seq = torch.cat(jet_seq, dim=0)
gen_jet_bin = gpt2.synthetic_jets.tokens_to_bins(jet_seq) 


data = SyntheticJets(shape_param=gpt2.shape, 
                    scale_param=gpt2.scale, 
                    bins_z= gpt2.bins_z,
                    bins_phi=gpt2.bins_phi,
                    z_order=True,
                    tokenize=True,
                    )

jet_seq  = data.sample(N=N)
jet_bin = data.tokens_to_bins(jet_seq)

kin_plots(jet_bin, gen_jet_bin, "res_kin.png")
ordered_z_plots(jet_bin, gen_jet_bin, "res_ord_z.png")


