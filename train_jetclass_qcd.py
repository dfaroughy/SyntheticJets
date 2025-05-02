# 1) Import Comet *before* torch/Lightning to capture all logs
import comet_ml
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader

from models import JetGPT2Model
from synthetic_data import JetSequenceDataset

##########################################################################
jet_type = 'ZJetsToNuNu'
num_nodes = 4
tags = [jet_type, 'perlmutter']
bins            = [41, 31, 31]
batch_size      = 128
n_embd          = 256
n_inner         = 1024
n_layer         = 8
n_head          = 4
lr              = 5e-4
lr_final        = 1e-6
max_epochs      = 16
##########################################################################

logger = CometLogger(
    api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz',
    project_name='tokenized-jets',
    workspace='dfaroughy',
    save_dir='/pscratch/sd/d/dfarough'
)

logger.experiment.add_tags(tags)

train_dataset = JetSequenceDataset(
    filepath=f"/pscratch/sd/d/dfarough/tokenized-jets/{jet_type}_train___1Mfromeach_403030.h5",
)

val_dataset = JetSequenceDataset(
    filepath=f"/pscratch/sd/d/dfarough/tokenized-jets/{jet_type}_val___1Mfromeach_403030.h5",
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

model = JetGPT2Model(
    n_embd=n_embd,
    n_inner=n_inner,
    n_layer=n_layer,
    n_head=n_head,
    learning_rate=lr,
)

trainer = L.Trainer(
    max_epochs=max_epochs,
    accelerator='gpu',
    devices='auto',
    strategy='ddp',
    num_nodes=num_nodes,
    callbacks=[
        L.callbacks.ModelCheckpoint(
            dirpath=None,
            monitor="val_loss",
            filename="best",
            save_top_k=1,
            mode="min",
            save_last=True,
        )
    ],
    logger=logger,
    sync_batchnorm=True,
    gradient_clip_val=1.0,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)