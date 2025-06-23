# import comet_ml
# import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from models import JetGPT2Model
from datamodule_jetclass import JetSequenceDataset

##########################################################################
parser = ArgumentParser()
parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
parser.add_argument("--project_name", "-proj", type=str, default='tokenized-jets')
parser.add_argument("--comet_workspace", type=str, default='dfaroughy')
parser.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
parser.add_argument("--data_path", type=str, default='/pscratch/sd/d/dfarough/JetClass')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--checkpoint", "-ckpt", type=str, default='last')
parser.add_argument("--tags", type=str, nargs='*')

parser.add_argument("--jet_type", "-type", type=str, default='ZJetsToNuNu')
parser.add_argument("--max_seq_length", "-len", type=int, default=40)
parser.add_argument("--num_bins", "-bins", type=int, nargs=3, default=[41, 31, 31])
parser.add_argument("--log_pt_range", "-pt", type=float, nargs=2, default=[-0.7602971186041831, 6.906254768371582])
parser.add_argument("--eta_range", "-eta", type=float, nargs=2, default=[-0.8, 0.8])
parser.add_argument("--phi_range", "-phi", type=float, nargs=2, default=[-0.8, 0.8])
parser.add_argument("--batch_size", "-bs", type=int, default=256)

parser.add_argument("--n_emb", type=int, default=256)
parser.add_argument("--n_inner", type=int, default=1024)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--pos_encoding", "-pos", type=bool, default=True)
parser.add_argument("--activation", "-a", type=str, default='gelu_new')
parser.add_argument("--dropout_attention", "-do_att", type=float, default=0.1)
parser.add_argument("--dropout_embedding", "-do_emb",type=float, default=0.1)
parser.add_argument("--dropout_residual", "-do_res", type=float, default=0.1)

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_final", type=float, default=0.0001)
parser.add_argument("--max_epochs", "-epochs", type=int, default=50)

config = parser.parse_args()
##########################################################################

logger = CometLogger(
            api_key=config.comet_api_key,
            project_name=config.project_name,
            workspace=config.comet_workspace,
            save_dir=config.dir,
            experiment_key=config.experiment_id if config.experiment_id else None
        )

logger.experiment.add_tags(config.tags)

train_dataset = JetSequenceDataset(filepath=f"{config.data_path}/train_100M_binned_{config.num_bins[0]-1}_{config.num_bins[1]-1}_{config.num_bins[2]-1}/train_{config.jet_type}_10M_binned.h5", 
                                   max_seq_length=config.max_seq_length,
                                   )

val_dataset = JetSequenceDataset(filepath=f"{config.data_path}/val_5M_binned_{config.num_bins[0]-1}_{config.num_bins[1]-1}_{config.num_bins[2]-1}/val_{config.jet_type}_500K_binned.h5", 
                                 max_seq_length=config.max_seq_length
                                 )

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)

checkpoint_every_epoch = L.callbacks.ModelCheckpoint(
    dirpath=None,
    filename="{epoch:03d}",       
    save_top_k=-1,                   # keep *all* checkpoints
    every_n_epochs=1,                # save after each epoch
    save_on_train_epoch_end=True,    # use training-epoch boundary
)

checkpoint_best_and_last = L.callbacks.ModelCheckpoint(
    dirpath=None,
    filename="best",                      # still keep the best model
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
)

trainer = L.Trainer(
    max_epochs=config.max_epochs,
    accelerator='gpu',
    devices='auto',
    strategy='ddp',
    num_nodes=config.num_nodes,
    callbacks=[checkpoint_every_epoch, checkpoint_best_and_last],
    logger=logger,
    sync_batchnorm=True,
    gradient_clip_val=1.0,
)


if config.experiment_id is None:

    model = JetGPT2Model(max_seq_length=config.max_seq_length,
                         num_bins=config.num_bins,
                         logpt_range=config.log_pt_range,
                         eta_range=config.eta_range,
                         phi_range=config.phi_range, 
                         n_embd=config.n_emb,
                         n_inner=config.n_inner,
                         n_layer=config.n_layer,
                         n_head=config.n_head,
                         activation=config.activation,
                         dropout_att=config.dropout_attention,
                         dropout_emb=config.dropout_embedding,
                         dropout_res=config.dropout_residual,
                         learning_rate=config.lr,
                         learning_rate_final=config.lr_final,
                         pos_encoding=config.pos_encoding,
                        )
    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
else:

    ckpt = f"{config.dir}/{config.project_name}/{config.experiment_id}/checkpoints/{config.checkpoint}.ckpt"
    model = JetGPT2Model.load_from_checkpoint(ckpt)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt
                )