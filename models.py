import numpy as np
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Config
from synthetic_data import SyntheticJets


class JetGPT2Model(L.LightningModule):
    def __init__(
        self,
        seq_length: int = 200,
        bins: list = [41, 31, 31],
        n_embd=128,
        n_layer=2,
        n_head=1,
        learning_rate=5e-4,
    ):
        super().__init__()

        # basic config
        self.seq_length = seq_length    # real tokens per jet
        self.bins = bins
        self.vocab_size = bins[0] * bins[1] * bins[2]
        self.lr = learning_rate
        self.do_sample = True  # sample multinomial
        self.temperature = 1.0
        self.top_k = None

        # special IDs
        self.start_token = self.vocab_size + 1
        self.end_token = self.vocab_size + 2
        self.pad_token = self.vocab_size + 3       

        config = GPT2Config(
            vocab_size=self.vocab_size + 4, # includes start/end/pad tokens in vocab
            n_positions=seq_length + 1,
            n_ctx=seq_length + 1,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=self.start_token,
            eos_token_id=self.end_token,
            pad_token_id=self.pad_token,
        )

        self.model = GPT2LMHeadModel(config)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return outputs.logits

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=self._mask_pads(batch["input_ids"]),
        )

        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def _mask_pads(self, labels):
        """
        Mask out the padding tokens in the labels.
        """
        labels = labels.clone()
        pads_mask = labels == self.pad_token
        labels[pads_mask] = -100
        return labels


    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=self._mask_pads(batch["input_ids"]),
        )

        loss = outputs.loss
        
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        seq = self.model.generate(
            input_ids=batch,
            max_new_tokens=self.seq_length, 
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k = self.top_k,
            bos_token_id=self.start_token,
            eos_token_id=self.end_token,
            pad_token_id=self.pad_token,
        )
        seq = seq.detach().cpu()
        return  seq # rm start token

        # results = []
        # for seq in generated_seq:
        #     seq = seq[1:] # rm start token
        #     idx_end_token = (seq == self.end_token).nonzero(as_tuple=True)[0]
        #     if idx_end_token.numel() > 0:
        #         seq = seq[: idx_end_token[0]]
        #     results.append(seq.detach().cpu())
        # return results

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def log_probs(self, sample, batch_size=256, device="cuda"):
        self.model.eval().to(device)
        N = sample.size(0)

        # prepend BOS
        bos = torch.full((N, 1), self.start_token, dtype=torch.long, device=device)
        seqs = torch.cat([bos, sample.to(device)], dim=1)

        ds = TensorDataset(seqs, seqs)
        loader = DataLoader(ds, batch_size=batch_size)

        all_logp = []
        for inp, lbl in loader:
            inp, lbl = inp.to(device), lbl.to(device)
            labels = self._mask_labels(lbl)
            outputs = self.model(input_ids=inp, labels=labels)
            # get logits for all but last
            logits = outputs.logits[:, :-1, :]
            target = inp[:, 1:]
            # compute tokenwise log-probs
            logp = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction="none",
            )
            # sum per sequence
            logp = logp.reshape(inp.size(0), -1).sum(dim=1)
            all_logp.append(logp.cpu())

        return torch.cat(all_logp, dim=0)

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











class SyntheticJetGPT2Model(L.LightningModule):
    def __init__(
        self,
        seq_length=30,
        vocab_size=None,
        n_embd=128,
        n_layer=2,
        n_head=1,
        pos_encoding=True,
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

        self.seq_length = seq_length
        self.lr = learning_rate
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.shape = shape_param
        self.scale = scale_param
        self.bins_z = bins_z
        self.bins_phi = bins_phi
        self.do_sample = True  # sample multinomial
        self.temperature = 1.0
        self.top_k = None
        self.pos_encoding = pos_encoding

        self.synthetic_jets = SyntheticJets(
            shape_param=shape_param,
            scale_param=scale_param,
            tokenize=True,
            z_order=True,
            bins_z=bins_z,
            bins_phi=bins_phi,
        )

        self.start_token = vocab_size # new token index for the start token

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
            eos_token_id=self.start_token
        )

        self.model = GPT2LMHeadModel(config)

        if not self.pos_encoding: # remove the learned positional embeddings
            with torch.no_grad():
                self.model.transformer.wpe.weight.zero_()
            self.model.transformer.wpe.weight.requires_grad = False

        self.save_hyperparameters()

    # ...training functions

    def forward(self, seq):
        outputs = self.model(seq)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, labels=batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
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
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss}

    # ...inference

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """outputs a jet sequence and the corresponding binned jet
        """
        seq = self.model.generate(
            input_ids=batch,
            max_length=self.seq_length + 1,  # total seq length
            do_sample=self.do_sample,    
            temperature=self.temperature,
            top_k = self.top_k,
            bos_token_id=self.start_token,
            pad_token_id=self.start_token + 1,  # avoid warning
        )

        seq = seq.detach().cpu()
        return  seq[:, 1:]  # rm start token

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # ...other functions

    @torch.no_grad()
    def log_probs(self, sample, batch_size=256, device="cuda"):
        """
        Compute total log-likelihoods for a batch of sequences.
        Args: seq (N, seq_length)
        Returns: log_probs (N,) — log-prob of each sequence
        """
        self.model.eval()
        self.model.to(device)
        N = sample.shape[0]

        start = torch.full((N, 1), self.start_token, dtype=torch.long, device=device)
        seqs = torch.cat([start, sample.to(device)], dim=1)  # (N, seq_len+1)
        dataset = TensorDataset(seqs)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        log_probs = []

        for (batch,) in dataloader:

            outputs = self.model(batch, labels=batch)
            logits = outputs.logits[:, :-1]       # (N, seq_len) shifted right
            labels = batch[:, 1:]                 # (N, seq_len)

            logp = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction='none'
            )

            logp = logp.reshape(batch.size(0), -1).sum(dim=-1)  # (N,)
            log_probs.append(logp.cpu())

        return torch.cat(log_probs, dim=0)

    @torch.no_grad()
    def per_token_preds(self, seq, device=None):
        """
        Compute per-token log p(x_t | x_{<t}) for a single sequence.
        Args:  seq (seq_length,)
        Returns: preds, log_probs (seq_length,)
        """
        self.model.eval()
        self.model.to(device)
        
        start_token = torch.tensor([self.start_token], device=device)
        seq = torch.cat([start_token, seq.to(device)])  # shape (seq_len+1,)

        input_tokens = seq[:-1].unsqueeze(0)  # (1, seq_len) shifted right
        target_tokens = seq[1:].unsqueeze(0)  # (1, seq_len)

        logits = self.model(input_tokens).logits   # (1, seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1))
        token_log_probs = token_log_probs.squeeze(0).squeeze(-1)  # (seq_len,)
        
        preds = torch.argmax(logits, dim=-1).squeeze(0)  # (seq_len,) greedy

        return preds.cpu(), token_log_probs.cpu()
