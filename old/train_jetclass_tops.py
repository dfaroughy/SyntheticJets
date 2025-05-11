# 1) Import Comet *before* torch/Lightning to capture all logs
import comet_ml
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader

from models import JetGPT2Model
from synthetic_data import JetSequenceDataset

##########################################################################
experiment_key  = None # '17eab02a204343a3969183034847009d'
jet_type = 'TTBar'
num_nodes = 4
tags = [jet_type, 'perlmutter', '10Mjets']
bins            = [41, 31, 31]
batch_size      = 128
n_embd          = 256
n_inner         = 1024
n_layer         = 8
n_head          = 4
lr              = 5e-4
lr_final        = 1e-6
max_epochs      = 50
##########################################################################

if experiment_key is None:
    logger = CometLogger(
        api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz',
        project_name='tokenized-jets',
        workspace='dfaroughy',
        save_dir='/pscratch/sd/d/dfarough'
    )
else:
    logger = CometLogger(
            api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz',
            project_name='tokenized-jets',
            workspace='dfaroughy',
            save_dir='/pscratch/sd/d/dfarough',
            experiment_key=experiment_key
        )

logger.experiment.add_tags(tags)

train_dataset = JetSequenceDataset(
    filepath=f"/pscratch/sd/d/dfarough/JetClass/train_100M_binned/train_{jet_type}_10M_bins403030.h5",
)

val_dataset = JetSequenceDataset(
    filepath=f"/pscratch/sd/d/dfarough/tokenized-jets/val_{jet_type}_500K_bins403030.h5",
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=32)

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


if experiment_key is None:

    model = JetGPT2Model(
                n_embd=n_embd,
                n_inner=n_inner,
                n_layer=n_layer,
                n_head=n_head,
                learning_rate=lr,
            )

    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
else:
    ckpt = f"/pscratch/sd/d/dfarough/tokenized-jets/{experiment_key}/checkpoints/last.ckpt"   
    model = JetGPT2Model.load_from_checkpoint(
                ckpt,
                n_embd=n_embd,
                n_inner=n_inner,
                n_layer=n_layer,
                n_head=n_head,
                learning_rate=lr,
            )
            
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt
                )