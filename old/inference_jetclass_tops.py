import numpy as np
import torch
import pytorch_lightning as L

from torch.utils.data import DataLoader
from models import JetGPT2Model

###############################################################################
N = 1000000   # num gen jets
top_k = 5000
tops_id = "17eab02a204343a3969183034847009d"        # comet run folder
path = "/pscratch/sd/d/dfarough/tokenized-jets/"
###############################################################################

tops_gpt2 = JetGPT2Model.load_from_checkpoint(f"{path}/{tops_id}/checkpoints/best.ckpt")
tops_gpt2.top_k = top_k

prompts = torch.full((N, 1), tops_gpt2.start_token, dtype=torch.long, device=tops_gpt2.device)
prompt_dataloadeer = DataLoader(prompts, batch_size=1024, shuffle=False)

generator = L.Trainer(accelerator="gpu", devices=[0])
tops_gen_seq = generator.predict(tops_gpt2, dataloaders=prompt_dataloadeer)
tops_gen_seq = torch.cat(tops_gen_seq, dim=0)

np.save(f'{path}{tops_id}/gen_tops_seq_gpt2_top{top_k}_jets{N}_25epochs.npy', tops_gen_seq)
