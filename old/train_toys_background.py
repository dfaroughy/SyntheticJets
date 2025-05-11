import numpy as np
import pytorch_lightning as L
from models import JetGPT2Model
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader, random_split

from synthetic_data import JetSequenceDataset

###############################################################################
tags = ['small_model', 'background', 'perlmutter']
num_jets = 500_000
num_constituents = 30          
shape = 1.0             # Gamma prior shape parameter.
scale = 1.0             # Gaussian prior scale parameter.

bins_z = np.linspace(0, 1, 21)
bins_phi = np.linspace(-15, 15, 31)  
split_train_val = 0.8
batch_size = 512
n_embd = 32
n_layer = 1
n_head = 1
pos_encoding = False
lr = 2.0e-4
max_epochs = 500
logger = CometLogger(api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz', 
                     project_name='synthetic-jets',
                     workspace='dfaroughy', 
                     save_dir='/pscratch/sd/d/dfarough/'
)

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

gpt2 = JetGPT2Model(
                seq_length=num_constituents,
                vocab_size=dataset.vocab_size, 
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                pos_encoding=pos_encoding,
                learning_rate=lr,
                shape_param=shape,
                scale_param=scale,
                bins_z=bins_z,
                bins_phi=bins_phi,
                )

logger.experiment.add_tags(tags)

trainer = L.Trainer(max_epochs=max_epochs, 
                    accelerator='gpu', 
                    devices=[0,1,2,3], 
                    strategy='ddp',
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
                    num_nodes=1,
                    )

trainer.fit(gpt2, train_dataloader, val_dataloader)