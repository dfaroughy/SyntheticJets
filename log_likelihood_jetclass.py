import numpy as np
import torch
import pytorch_lightning as L
from argparse import ArgumentParser

from torch.utils.data import DataLoader, TensorDataset
from models import JetGPT2Model
from utils import LogProbsCallback
from datamodule_jetclass import JetSequenceDataset


###############################################################################
parser = ArgumentParser()

parser.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
parser.add_argument("--predict_type", type=str, default='logp')
parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--project_name", "-proj", type=str, default='tokenized-jets')
parser.add_argument("--tag", type=str, default=None)

parser.add_argument("--jet_type", type=str, default='TTBar')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--eval_data_type", "-eval", type=str, default='TTBar')
parser.add_argument("--eval_data_path", "-eval_pth", type=str, default=None)
parser.add_argument("--num_jets", "-n", type=int, default=10000)
parser.add_argument("--batch_size", "-bs", type=int, default=128)

config = parser.parse_args()

###############################################################################



model = JetGPT2Model.load_from_checkpoint(f"{config.dir}/{config.project_name}/{config.experiment_id}/checkpoints/best.ckpt", map_location="cpu",)
model.predict_type = config.predict_type

seq = torch.tensor(np.load(f"{config.dir}/{config.project_name}/{config.eval_data_path}"))
dataset = JetSequenceDataset(input_ids=seq, num_jets=config.num_jets)
dataloader = DataLoader(dataset, batch_size=config.batch_size)

callback = LogProbsCallback(config=config)


#...compute log probs

comp_log_probs = L.Trainer(accelerator="gpu", 
                          devices='auto', 
                          strategy='ddp', 
                          num_nodes=config.num_nodes, 
                          callbacks=[callback])

comp_log_probs.predict(model, dataloaders=dataloader)

