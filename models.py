import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from scipy.special import gammaln, gamma, factorial

from transformers import GPT2LMHeadModel, GPT2Config
from datamodule_jetclass import JetSequence

class JetGPT2Model(L.LightningModule):
    def __init__(
        self,
        max_seq_length: int = 40,
        logpt_range=(-0.7602971186041831, 6.906254768371582),
        eta_range=(-0.8, 0.8),
        phi_range=(-0.8, 0.8), 
        num_bins=[40, 30, 30],
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
        self.num_bins = num_bins
        self.logpt_range = logpt_range
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.vocab_size = num_bins[0] * num_bins[1] * num_bins[2]

        # model config
        self.lr = learning_rate
        self.lr_final = learning_rate_final
        self.do_sample = True  # sample multinomial
        self.temperature = temperature
        self.top_k = top_k
        self.pos_encoding = pos_encoding

        # volume element:

        dpt = np.abs(logpt_range[1] - logpt_range[0]) / num_bins[0] 
        deta = np.abs(eta_range[1] - eta_range[0]) / num_bins[1]
        dphi = np.abs(phi_range[1] - phi_range[0]) / num_bins[2]
        self.dvol = dpt * deta * dphi  # log of the volume element 

        print('INFO: bins:', num_bins)
        print('INFO: vocab size:', self.vocab_size)
        print('INFO: vol element dV=', self.dvol)
        # self.log_dvol= -7.421279091726184, 

        # special tokens:

        self.start_token = self.vocab_size + 1
        self.end_token = self.vocab_size + 2
        self.pad_token = self.vocab_size + 3  

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
        self.predict_type = 'gen' # 'gen' or 'logp' modes

        if not self.pos_encoding:
            # If pos_encoding is disabled, zero & freeze GPT-2's pos embeddings
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

        if self.predict_type == 'gen':

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

        elif self.predict_type == 'logp':

            preds = self.compute_log_probs(batch, include_symmetry_terms=True) 

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
    def compute_log_probs(self, batch, symmetry_correction_terms=True):

        self.model.eval()
        
        batch_ids  = batch["input_ids"] 
        batch_mask =  batch["attention_mask"] 

        targets = batch_ids.clone()
        targets[targets == self.pad_token] = -100  # for CE ignore_index

        outputs = self.model(input_ids=batch_ids,
                             attention_mask=batch_mask,
                            )

        logits = outputs.logits[:, :-1] # drop last token pred
        targets = targets[:, 1:]  # align labels by shifting right

        logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1),
                                reduction="none",
                                ignore_index=-100, # ignore pad-tokens
                                ).view(batch_ids.size(0), -1)

        logp = logp.sum(dim=1)

        if symmetry_correction_terms:

            jet_seq = JetSequence(max_seq_length=self.max_seq_length,
                                  start_token=self.start_token,
                                  end_token= self.end_token,
                                  pad_token=self.pad_token,
                                  num_bins=self.num_bins) 

            N = jet_seq.multiplicities(batch_ids.cpu().numpy())
            logp_sym = torch.log(torch.tensor(factorial(N)))
            logp_repeats = jet_seq.log_symmetry_factor(batch_ids.cpu().numpy())
            logp -= torch.tensor(logp_sym, device=logp.device, dtype=logp.dtype)   # - log(N!)
            logp += torch.tensor(logp_repeats, device=logp.device, dtype=logp.dtype) # sum_k log(N_k!) where N_k is multiplicity of k-th pt bin
            logp -= torch.tensor( N * np.log(self.dvol), device=logp.device, dtype=logp.dtype) # - N * log(dvol) volume factor

        return logp


    @torch.no_grad()
    def per_token_preds(self, seq, device=None):

        self.model.eval().to(device)

        seq = seq.to(device)
        inp = seq[:-1].unsqueeze(0)
        mask = (seq[:-1] != self.pad_token).unsqueeze(0).long()
        tgt = seq[1:].unsqueeze(0)

        logits = self.model(input_ids=inp, 
                            attention_mask=mask).logits
                            
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)
        preds = logits.argmax(dim=-1).squeeze(0)

        return preds.cpu(), token_log_probs.cpu()

    def _mask_pads(self, labels):
        """ Mask out the padding tokens in the labels.
        """
        labels = labels.clone()
        labels[labels == self.pad_token] = -100  # CE ignores
        return labels
