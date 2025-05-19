import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Config
from datamodule_synthetic_jets import SyntheticJets


class JetClassGPTModel(L.LightningModule):
    def __init__(
        self,
        max_seq_length: int = 200,
        bins: list = [41, 31, 31],
        n_embd=128,
        n_inner=None,
        n_layer=2,
        n_head=1,
        activation='gelu_new',
        dropout_att=0.1,
        dropout_emb=0.1,
        dropout_res=0.1,
        learning_rate=5e-4,
        learning_rate_final=0.0,
        top_k=None,
        temperature=1.0,
        pos_encoding=True,
    ):
        super().__init__()

        # basic config
        self.max_seq_length = max_seq_length    # real tokens per jet
        self.bins = bins
        self.vocab_size = bins[0] * bins[1] * bins[2]
        self.lr = learning_rate
        self.lr_final = learning_rate_final
        self.do_sample = True  # sample multinomial
        self.temperature = temperature
        self.top_k = top_k
        self.pos_encoding = pos_encoding

        # special IDs
        self.start_token = self.vocab_size
        self.end_token = self.vocab_size + 1
        self.pad_token = self.vocab_size + 2       

        config = GPT2Config(
            vocab_size=self.vocab_size + 3, # token vocab + BOS + EOS + pads
            n_positions=max_seq_length + 2, # seq with BOS and EOS enpoints
            n_ctx=max_seq_length + 2, # seq with BOS and EOS enpoints
            n_embd=n_embd,
            n_inner=n_inner if n_inner is not None else 4 * n_embd,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=activation,
            attn_pdrop=dropout_att,
            embd_pdrop=dropout_emb,
            resid_pdrop=dropout_res,
            bos_token_id=self.start_token,
            eos_token_id=self.end_token,
            pad_token_id=self.pad_token,
        )

        self.model = GPT2LMHeadModel(config)
        self.model.predict_type = 'gen'

        # If pos_encoding is disabled, zero & freeze GPT-2's position embeddings:

        if not self.pos_encoding:
            with torch.no_grad():
                self.model.transformer.wpe.weight.zero_()
            self.model.transformer.wpe.weight.requires_grad = False

        self.save_hyperparameters()

    #...train/inference methods

    def forward(self, input_ids, attention_mask=None):

        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask
                             )
        
        return outputs.logits

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=self._mask_pads(batch["input_ids"]),
                            )

        loss = outputs.loss
        self.log("train_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=self._mask_pads(batch["input_ids"]),
                             )
        loss = outputs.loss
        self.log("val_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 )
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        if self.model.predict_type == 'gen':

            preds = self.model.generate(
                            input_ids=batch,
                            max_new_tokens=self.max_seq_length + 2, 
                            do_sample=self.do_sample,
                            temperature=self.temperature,
                            top_k = self.top_k,
                            bos_token_id=self.start_token,
                            eos_token_id=self.end_token,
                            pad_token_id=self.pad_token,
                        )

            preds = F.pad(preds, (0, self.max_seq_length + 2 - preds.shape[1]), value=self.pad_token)

        elif self.model.predict_type == 'logp':
            
            preds = self.compute_log_probs(batch) 

        return preds.detach().cpu()
 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,    # full cycle length
            eta_min=self.lr_final             # final LR
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   
                "frequency": 1,
                "strict": True,
            },
        }

    #...prediction funcs

    @torch.no_grad()
    def compute_log_probs(self, batch, preprocessed=False):

        labels = batch["input_ids"].clone()
        labels[labels >= self.start_token] = -100 

        outputs = self.model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=labels,
                            )

        logits = outputs.logits[:, :-1, :] # drop end token pred
        labels = labels[:, 1:] # align labels by shifting right

        logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                labels.reshape(-1),
                                reduction="none",
                                ignore_index=-100,
                                )

        return logp.reshape(batch["input_ids"].size(0), -1).sum(dim=1)  # (B,)

    @torch.no_grad()
    def per_token_preds(self, seq, device=None):
        self.model.eval().to(device)

        # prepend BOS
        seq = torch.cat([torch.tensor([self.start_token], device=device), seq.to(device)])
        inp = seq[:-1].unsqueeze(0)
        tgt = seq[1:].unsqueeze(0)

        logits = self.model(input_ids=inp).logits
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)
        preds = logits.argmax(dim=-1).squeeze(0)

        return preds.cpu(), token_log_probs.cpu()

    def _mask_pads(self, labels):
        """ Mask out the padding tokens in the labels.
        """
        labels = labels.clone()
        pads_mask = labels >= self.pad_token
        labels[pads_mask] = -100  # CE ignores
        return labels

    def _seq_pad(self, sample):    
        
        results = []
        
        for seq in sample:
            if seq.numel() < self.max_seq_length:
                seq = F.pad(seq, (0, self.max_seq_length - seq.numel()), value=self.pad_token)
            else:
                seq = seq[: self.max_seq_length]
                
            results.append(seq)
        
        return torch.stack(results)