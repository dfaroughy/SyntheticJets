import numpy as np
import pytorch_lightning as L
from models import GPT2Model
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader, random_split

from synthetic_data import JetSequenceDataset

###############################################################################
tags = ['qcd', 'coarse', 'pascal2']
num_jets = 10000
num_constituents = 30          
shape = 1.0             # Gamma prior shape parameter.
scale = 1.0             # Gaussian prior scale parameter.
bins_z = np.linspace(0, 1, 11)
bins_phi = np.linspace(-15, 15, 16)  
split_train_val = 0.8
batch_size = 512
n_embd = 128
n_layer = 2
n_head = 2
lr = 1e-3
max_epochs = 100
logger = CometLogger(api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz', 
                     project_name='synthetic-jets',
                     workspace='dfaroughy', 
                     save_dir='/home/df630/SyntheticJets/experiments/results/comet')

###############################################################################


dataset = JetSequenceDataset(num_samples=num_jets,
                            num_constituents= num_constituents,
                            shape_param= shape,
                            scale_param= scale,
                            bins_z = bins_z,
                            bins_phi= bins_phi,
                            )

train_size = int(split_train_val * len(dataset))
val_size   = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

gpt2 = GPT2Model(
                seq_length=num_constituents,
                vocab_size=dataset.vocab_size, 
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                learning_rate=lr,
                shape_param=shape,
                scale_param=scale,
                bins_z=bins_z,
                bins_phi=bins_phi,
                )

logger.experiment.add_tags(tags)

trainer = L.Trainer(max_epochs=max_epochs, 
                    accelerator='gpu', 
                    devices=[0,2,3], 
                    strategy='auto',
                    callbacks=[L.callbacks.ModelCheckpoint(dirpath=None,
                                                           monitor="val_loss",
                                                           filename="best",
                                                           save_top_k=1,
                                                           mode="min",
                                                           save_last=True,
                                                            )],
                    logger=logger,
                    sync_batchnorm=True,
                    gradient_clip_val=1.0,
                    )

trainer.fit(gpt2, train_dataloader, val_dataloader)