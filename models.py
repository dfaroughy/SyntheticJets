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
        log_dvol=-7.4212790,  # dvol = dpt * deta * dphi volume element
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

        self.log_dvol = log_dvol       

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
        self.predict_type = 'gen'

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
    def compute_log_probs(self, batch, include_symmetry_terms=False):

        self.model.eval()
        
        batch_ids  = batch["input_ids"]#.squeeze(1)
        batch_mask =  batch["attention_mask"]#.squeeze(1)

        targets = batch_ids.clone()
        targets[targets == self.pad_token] = -100  # CE ignores

        outputs = self.model(input_ids=batch_ids,
                             attention_mask=batch_mask,
                            )

        logits = outputs.logits[:, :-1] # drop last token pred
        targets = targets[:, 1:]  # align labels by shifting right

        logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1),
                                reduction="none",
                                ignore_index=-100,
                                ).view(batch_ids.size(0), -1)

        logp = logp.sum(dim=1)

        if include_symmetry_terms:  # add log(N!):

            jet_seq = JetSequence(start_token=self.start_token,
                                  end_token= self.end_token,
                                  pad_token=self.pad_token,) 

            N = jet_seq.multiplicities(batch_ids.cpu().numpy())
            logp_sym = torch.log(torch.tensor(factorial(N)))
            logp_repeats = jet_seq.log_symmetry_factor(batch_ids.cpu().numpy())
            logp -= torch.tensor(logp_sym, device=logp.device, dtype=logp.dtype)   # - log(N!)
            logp += torch.tensor(logp_repeats, device=logp.device, dtype=logp.dtype) # sum_k log(N_k!) where N_k is multiplicity of k-th pt bin
            logp -= torch.tensor( N * (self.log_dvol), device=logp.device, dtype=logp.dtype) # - N * log(dvol) volume factor

        return logp


    @torch.no_grad()
    def per_token_preds(self, seq, device=None):
        self.model.eval().to(device)

        # prepend BOS
        # seq = torch.cat([torch.tensor([self.start_token], device=device), seq.to(device)])
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

    # def _seq_pad(self, sample):    
        
    #     results = []
        
    #     for seq in sample:
    #         if seq.numel() < self.max_seq_length:
    #             seq = F.pad(seq, (0, self.max_seq_length - seq.numel()), value=self.pad_token)
    #         else:
    #             seq = seq[: self.max_seq_length]
                
    #         results.append(seq)
        
    #     return torch.stack(results)

    # @torch.no_grad()
    # def log_probs(self, sample, batch_size=256, device="cuda"):
    #     ''' WARNING: input sample need to be preprocessed. 
    #         i.e. with no end/start tokens and pads == -1
    #     '''

    #     self.model.eval()
    #     self.model.to(device)

    #     seq = sample.to(device)
    #     attn_mask = (seq != -1).long()
    #     dataset = TensorDataset(seq, attn_mask)
    #     dataloader = DataLoader(dataset, batch_size=batch_size)
    #     log_probs = []

    #     for batch_ids, batch_mask in dataloader: 
    #         batch_ids = batch_ids.to(device)
    #         batch_mask = batch_mask.to(device)
    #         logp = self._log_probs(batch_ids, batch_mask, preprocessed=True)
    #         log_probs.append(logp.cpu())

    #     return torch.cat(log_probs, dim=0)


# class SyntheticJetGPT2Model(L.LightningModule):
#     def __init__(
#         self,
#         seq_length=30,
#         vocab_size=None,
#         n_embd=128,
#         n_layer=2,
#         n_head=1,
#         pos_encoding=True,
#         learning_rate=5e-4,
#         shape_param=1.0,
#         scale_param=1.0,
#         bins_z=None,
#         bins_phi=None,
#     ):
#         """
#         Initializes a GPT-2 model for next-token prediction on synthetic jet token sequences.
#         A special start token is prepended to each sequence.
#         """

#         super().__init__()

#         self.seq_length = seq_length
#         self.lr = learning_rate
#         self.n_embd = n_embd
#         self.n_layer = n_layer
#         self.n_head = n_head
#         self.shape = shape_param
#         self.scale = scale_param
#         self.bins_z = bins_z
#         self.bins_phi = bins_phi
#         self.do_sample = True  # sample multinomial
#         self.temperature = 1.0
#         self.top_k = None
#         self.pos_encoding = pos_encoding

#         self.synthetic_jets = SyntheticJets(
#             shape_param=shape_param,
#             scale_param=scale_param,
#             tokenize=True,
#             z_order=True,
#             bins_z=bins_z,
#             bins_phi=bins_phi,
#         )

#         self.start_token = vocab_size # new token index for the start token

#         print(
#             "INFO: vocab_size: {}, start_token_id: {}".format(
#                 vocab_size, self.start_token
#             )
#         )

#         # ...create GPT-2 configuration:

#         config = GPT2Config(
#             vocab_size=vocab_size + 1,  # includes the start token.
#             n_positions=seq_length + 1,  # includes the start token.
#             n_ctx=1,  # prompt length is 1
#             n_embd=n_embd,
#             n_layer=n_layer,
#             n_head=n_head,
#             bos_token_id=self.start_token,
#             eos_token_id=self.start_token
#         )

#         self.model = GPT2LMHeadModel(config)

#         if not self.pos_encoding: # remove the learned positional embeddings
#             with torch.no_grad():
#                 self.model.transformer.wpe.weight.zero_()
#             self.model.transformer.wpe.weight.requires_grad = False

#         self.save_hyperparameters()

#     # ...training functions

#     def forward(self, seq):
#         outputs = self.model(seq)
#         return outputs.logits

#     def training_step(self, batch, batch_idx):
#         outputs = self.model(batch, labels=batch)
#         loss = outputs.loss
#         self.log(
#             "train_loss",
#             loss,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#             sync_dist=True,
#         )
#         return {"loss": loss}

#     def validation_step(self, batch, batch_idx):
#         outputs = self.model(batch, labels=batch)
#         loss = outputs.loss
#         self.log(
#             "val_loss",
#             loss,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#             sync_dist=True,
#         )
#         return {"val_loss": loss}

#     # ...inference

#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         """outputs a jet sequence and the corresponding binned jet
#         """
#         seq = self.model.generate(
#                                 input_ids=batch,
#                                 max_length=self.seq_length + 1,  # total seq length
#                                 do_sample=self.do_sample,    
#                                 temperature=self.temperature,
#                                 top_k = self.top_k,
#                                 bos_token_id=self.start_token,
#                                 pad_token_id=self.start_token + 1,
#                                 synced_gpus=True
#                                 )

#         seq = seq.detach().cpu()

#         return  seq[:, 1:]  # rm start token

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=self.lr)

#     # ...other functions

    # @torch.no_grad()
    # def log_probs(self, sample, batch_size=256, device="cuda"):
    #     """
    #     Compute total log-likelihoods for a batch of sequences.
    #     Args: seq (N, seq_length)
    #     Returns: log_probs (N,) — log-prob of each sequence
    #     """
    #     self.model.eval()
    #     self.model.to(device)
    #     N = sample.shape[0]

    #     start = torch.full((N, 1), self.start_token, dtype=torch.long, device=device)
    #     seqs = torch.cat([start, sample.to(device)], dim=1)  # (N, seq_len+1)
    #     dataset = TensorDataset(seqs)
    #     dataloader = DataLoader(dataset, batch_size=batch_size)

    #     log_probs = []

    #     for (batch,) in dataloader:

    #         outputs = self.model(batch, labels=batch)
    #         logits = outputs.logits[:, :-1]       # (N, seq_len) shifted right
    #         labels = batch[:, 1:]                 # (N, seq_len)

    #         logp = -F.cross_entropy(
    #             logits.reshape(-1, logits.size(-1)),
    #             labels.reshape(-1),
    #             reduction='none'
    #         )

    #         logp = logp.reshape(batch.size(0), -1).sum(dim=-1)  # (N,)
    #         log_probs.append(logp.cpu())

    #     return torch.cat(log_probs, dim=0)

#     @torch.no_grad()
#     def per_token_preds(self, seq, device=None):
#         """
#         Compute per-token log p(x_t | x_{<t}) for a single sequence.
#         Args:  seq (seq_length,)
#         Returns: preds, log_probs (seq_length,)
#         """
#         self.model.eval()
#         self.model.to(device)
        
#         start_token = torch.tensor([self.start_token], device=device)
#         seq = torch.cat([start_token, seq.to(device)])  # shape (seq_len+1,)

#         input_tokens = seq[:-1].unsqueeze(0)  # (1, seq_len) shifted right
#         target_tokens = seq[1:].unsqueeze(0)  # (1, seq_len)

#         logits = self.model(input_tokens).logits   # (1, seq_len, vocab_size)
#         log_probs = F.log_softmax(logits, dim=-1)
#         token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1))
#         token_log_probs = token_log_probs.squeeze(0).squeeze(-1)  # (seq_len,)
        
#         preds = torch.argmax(logits, dim=-1).squeeze(0)  # (seq_len,) greedy

#         return preds.cpu(), token_log_probs.cpu()
