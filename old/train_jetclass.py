# 1) Import Comet *before* torch/Lightning to capture all logs
import comet_ml
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader

from models import JetGPT2Model
from synthetic_data import JetSequenceDataset

##########################################################################
tags = ['ZJetsToNuNu', 'pascal2']
bins            = [41, 31, 31]
batch_size      = 32
n_embd          = 128
n_layer         = 4
n_head          = 4
lr              = 1e-3
max_epochs      = 500
##########################################################################

logger = CometLogger(
    api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz',
    project_name='tokenized-jets',
    workspace='dfaroughy',
    save_dir='/home/df630/SyntheticJets/experiments/results/comet'
)
logger.experiment.add_tags(tags)

train_dataset = JetSequenceDataset(
    filepath="data/ZJetsToNuNu_train___1Mfromeach_403030.h5",
    num_jets=100_000,
)

val_dataset = JetSequenceDataset(
    filepath="data/ZJetsToNuNu_val___1Mfromeach_403030.h5",
    num_jets=10_000,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

model = JetGPT2Model(
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    learning_rate=lr,
)

trainer = L.Trainer(
    max_epochs=max_epochs,
    accelerator='gpu',
    devices=[0,1,2,3],
    strategy='auto',
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
    precision=16,
    sync_batchnorm=True,
    gradient_clip_val=1.0,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)