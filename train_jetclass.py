# import comet_ml
# import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from models import JetGPT2Model
from synthetic_data import JetSequenceDataset

##########################################################################
parser = ArgumentParser()
parser.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
parser.add_argument("--comet_workspace", type=str, default='dfaroughy')
parser.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
parser.add_argument("--project_name", "-proj", type=str, default='tokenized-jets')
parser.add_argument("--data_path", type=str, default='/pscratch/sd/d/dfarough/JetClass')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--tags", type=str, nargs='*')
parser.add_argument("--jet_type", "-type", type=str, default='ZJetsToNuNu')
parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--nBins", "-bins", type=int, nargs=3)
parser.add_argument("--batch_size", "-bs", type=int, default=128)
parser.add_argument("--n_emb", type=int, default=256)
parser.add_argument("--n_inner", type=int, default=256)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_final", type=float, default=0.00005)
parser.add_argument("--max_epochs", "-epochs", type=int, default=50)

args = parser.parse_args()
##########################################################################

logger = CometLogger(
            api_key=args.comet_api_key,
            project_name=args.project_name,
            workspace=args.comet_workspace,
            save_dir=args.dir
            experiment_key=args.experiment_id if args.experiment_id else None
        )

logger.experiment.add_tags(args.tags)

train_dataset = JetSequenceDataset(filepath=f"{args.data_path}/train_100M_binned/train_{args.jet_type}_10M_bins403030.h5")
val_dataset = JetSequenceDataset(filepath=f"{args.data_path}/val_5M_binned/val_{args.jet_type}_500K_bins403030.h5")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

trainer = L.Trainer(
    max_epochs=args.max_epochs,
    accelerator='gpu',
    devices='auto',
    strategy='ddp',
    num_nodes=args.num_nodes,
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


if args.experiment_id is None:

    model = JetGPT2Model(
                n_embd=args.n_emb,
                n_inner=args.n_inner,
                n_layer=args.n_layer,
                n_head=args.n_head,
                learning_rate=args.lr,
                learning_rate_final=args.lr_final
            )

    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
else:
    ckpt = f"{args.dir}/{args.project_name}/{args.experiment_id}/checkpoints/last.ckpt"
    model = JetGPT2Model.load_from_checkpoint(ckpt)
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt
                )