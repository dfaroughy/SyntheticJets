import numpy as np
import torch
import pytorch_lightning as L

from transformers import GPT2LMHeadModel, GPT2Config
from synthetic_data import SyntheticJets


class GPT2Model(L.LightningModule):
    def __init__(
        self,
        seq_length=30,
        vocab_size=None,
        n_embd=128,
        n_layer=2,
        n_head=1,
        learning_rate=5e-4,
        shape_param=1.0,
        scale_param=1.0,
        bins_z=None,
        bins_phi=None,
    ):
        """
        Initializes a GPT-2 model for next-token prediction on synthetic jet token sequences.
        A special start token is prepended to each sequence.
        """

        super().__init__()

        self.D = seq_length
        self.lr = learning_rate
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.shape = shape_param
        self.scale = scale_param
        self.bins_z = bins_z
        self.bins_phi = bins_phi
        self.predict_output = 'jet_sequence'

        self.jet_seq = SyntheticJets(
            shape_param=shape_param,
            scale_param=scale_param,
            tokenize=True,
            z_order=True,
            bins_z=bins_z,
            bins_phi=bins_phi,
        )

        self.start_token = vocab_size  # new token index for the start token

        print(
            "INFO: vocab_size: {}, start_token_id: {}".format(
                vocab_size, self.start_token
            )
        )

        # ...create GPT-2 configuration:

        config = GPT2Config(
            vocab_size=vocab_size + 1,  # includes the start token.
            n_positions=seq_length + 1,  # includes the start token.
            n_ctx=1,  # prompt length is 1
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=self.start_token,
        )

        self.model = GPT2LMHeadModel(config)
        self.save_hyperparameters()

    def forward(self, seq):
        outputs = self.model(seq)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, labels=batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, labels=batch)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss}

    #...inference

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seq = (
            self.model.generate(
                input_ids=batch,
                max_length=self.D + 1,  # total seq length
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.start_token + 1,  # avoid warning
            )
            .detach()
            .cpu()
        )

        if self.predict_output == 'jet_sequence':
            return seq[:, 1:]

        elif self.predict_output == 'jet_binned':
            jet_binned = self.jet_seq.token_to_bins(seq[:, 1:])  # rm start token
            return torch.from_numpy(jet_binned)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
