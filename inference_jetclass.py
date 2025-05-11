import numpy as np
import torch
import pytorch_lightning as L
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from models import JetGPT2Model

###############################################################################
parser = ArgumentParser()

parser.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
parser.add_argument("--project_name", "-proj", type=str, default='tokenized-jets')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--jet_type", "-type", type=str, default=None)
parser.add_argument("--top_k", type=int, default=5000)
parser.add_argument("--num_jets", "-n", type=int, default=1000000)
parser.add_argument("--batch_size", "-bs", type=int, default=1024)

args = parser.parse_args()
###############################################################################

model = JetGPT2Model.load_from_checkpoint(f"{args.dir}/{args.project_name}/{args.experiment_id}/checkpoints/best.ckpt")
model.top_k = args.top_k  

prompts = torch.full((args.num_jets, 1), model.start_token, dtype=torch.long, device=model.device)
prompt_dataloadeer = DataLoader(prompts, batch_size=args.batch_size, shuffle=False)

generator = L.Trainer(accelerator="gpu", devices=[0])
gen_seq = generator.predict(model, dataloaders=prompt_dataloadeer)
gen_seq = torch.cat(gen_seq, dim=0)

np.save(f'{args.dir}/{args.project_name}/{args.experiment_id}/gen_{args.jet_type}_seq_top{args.top_k}_jets{args.num_jets}.npy', gen_seq)