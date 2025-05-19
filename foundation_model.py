import torch, torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Config, GPT2DoubleHeadsModel

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Config
from datamodule_synthetic_jets import SyntheticJets


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
    ):
        super().__init__()

        # basic config
        self.max_seq_length = max_seq_length    # real tokens per jet
        self.bins = bins
        self.vocab_size_kin = bins[0] * bins[1] * bins[2]
        self.lr = learning_rate
        self.lr_final = learning_rate_final
        self.do_sample = True  # sample multinomial
        self.temperature = temperature
        self.top_k_kin = top_k_kin

        # special IDs
        self.start_token_kin = self.vocab_size_kin
        self.end_token_kin= self.vocab_size_kin+ 1
        self.pad_token_kin = self.vocab_size_kin + 2  

        self.start_token_flavor = self.vocab_size_flavor
        self.end_token_flavor= self.vocab_size_flavor + 1
        self.pad_token_flavor = self.vocab_size_flavor + 2 

        config_kin = GPT2Config(
            vocab_size=self.vocab_size_kin + 3, # token vocab + BOS + EOS + pads
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

        config_flavor = GPT2Config(
            vocab_size= self.vocab_size_flavor + 3, # token vocab + BOS + EOS + pads
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


       # self.wte_pt = nn.Emebeddings(n_embd, bins[0])
       # self.wte_eta = nn.Emebeddings(n_embd, bins[1])
       # self.wte_phi = nn.Emebeddings(n_embd, bins[2])

        self.model_kin = GPT2LMHeadModel(config_flavor)
        self.model_kin = GPT2LMHeadModel(config_kin)

        # If pos_encoding is disabled, zero & freeze GPT-2's position embeddings:

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

        gen_seq = self.model.generate(
                            input_ids=batch,
                            max_new_tokens=self.max_seq_length + 2, 
                            do_sample=self.do_sample,
                            temperature=self.temperature,
                            top_k = self.top_k,
                            bos_token_id=self.start_token,
                            eos_token_id=self.end_token,
                            pad_token_id=self.pad_token,
                        )

        results = []
        gen_seq = gen_seq.detach().cpu()
        
        # preprocess results:
        for seq in gen_seq:
            seq = seq[1:] # rm start token
            seq[seq == self.end_token] = -1 # replace end token with -1
            seq[seq == self.pad_token] = -1 # replace pad token with -1

            if seq.numel() < self.max_seq_length:
                seq = F.pad(seq, (0, self.max_seq_length - seq.numel()), value=-1)
            else:
                seq = seq[: self.max_seq_length]
                
            results.append(seq)
        
        return torch.stack(results) 


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

    #...log probs

    @torch.no_grad()
    def log_probs(self, sample, batch_size=256, device="cuda"):
        ''' WARNING: input sample need to be preprocessed. 
            i.e. with no end/start tokens and pads == -1
        '''

        self.model.eval()
        self.model.to(device)

        seq = sample.to(device)
        attn_mask = (seq != -1).long()
        dataset = TensorDataset(seq, attn_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        log_probs = []

        for batch_ids, batch_mask in dataloader: 
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            logp = self._log_probs(batch_ids, batch_mask, preprocessed=True)
            log_probs.append(logp.cpu())

        return torch.cat(log_probs, dim=0)

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

    #...helpers

    def _mask_pads(self, labels):
        """ Mask out the padding tokens in the labels.
        """
        labels = labels.clone()
        pads_mask = labels >= self.pad_token
        labels[pads_mask] = -100  # CE ignores
        return labels

    @torch.no_grad()
    def _log_probs(self, batch_ids, batch_mask, preprocessed=False):

        labels = batch_ids.clone()

        if preprocessed:
            labels[labels == -1]   = -100
        else:                
            labels[labels == self.start_token] = -100 
            labels[labels == self.pad_token]   = -100
            labels[labels == self.end_token]   = -100

        outputs = self.model(input_ids=batch_ids,
                             attention_mask=batch_mask,
                             labels=labels,
                            )

        logits = outputs.logits[:, :-1, :] # drop end token pred
        labels = labels[:, 1:] # align labels by shifting right

        logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                labels.reshape(-1),
                                reduction="none",
                                ignore_index=-100,
                                )

        return logp.reshape(batch_ids.size(0), -1).sum(dim=1)  # (B,)


class JetGPT2DoubleHeads(L.LightningModule):
    # ------------------------------ init & config ------------------------------
    def __init__(
        self,
        max_seq_length: int = 200,
        bins: tuple = (41, 31, 31),
        n_embd: int = 128,
        n_inner: int | None = None,
        n_layer: int = 2,
        n_head: int = 1,
        activation: str = "gelu_new",
        dropout_att: float = 0.1,
        dropout_emb: float = 0.1,
        dropout_res: float = 0.1,
        learning_rate: float = 5e-4,
        learning_rate_final: float = 0.0,
        mc_weight: float = 5.0,          #  λ  in  L = L_MC + λ·L_LM
        stop_threshold: float = 0.70,     #  θ  at sampling time
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.bins = bins
        self.vocab_size = bins[0] * bins[1] * bins[2]

                # basic config
        self.lr = learning_rate
        self.lr_final = learning_rate_final
        self.do_sample = True  # sample multinomial
        self.mc_weight      = mc_weight
        self.stop_threshold = stop_threshold
        self.temperature    = temperature
        self.top_k          = top_k
        self.MIN_CONSTITUENTS  = 3

        self.start_token = self.vocab_size + 1   # BOS
        self.end_token   = self.vocab_size + 2   # EOS
        self.pad_token   = self.vocab_size + 3   # PAD
        self.cls_token   = self.vocab_size + 4   # CLS (MC anchor)

        cfg = GPT2Config(
            vocab_size   = self.vocab_size + 5,
            n_positions  = max_seq_length + 2,   # room for CLS
            n_ctx        = max_seq_length + 2,
            n_embd       = n_embd,
            n_inner      = n_inner or 4 * n_embd,
            n_layer      = n_layer,
            n_head       = n_head,
            activation_function = activation,
            attn_pdrop   = dropout_att,
            embd_pdrop   = dropout_emb,
            resid_pdrop  = dropout_res,
            bos_token_id = self.start_token,
            eos_token_id = self.end_token,
            pad_token_id = self.pad_token,
        )

        self.model = GPT2DoubleHeadsModel(cfg)

        self.save_hyperparameters()


    def training_step(self, batch, _):
        mc = self._build_mc_batch(batch["input_ids"])

        lm_labels = self._mask_pads(mc["mc_input_ids"])
        out = self.model(
            input_ids      = mc["mc_input_ids"],
            attention_mask = mc["attention"],
            mc_token_ids   = mc["mc_token_ids"],
            mc_labels      = mc["mc_labels"],
            labels         = lm_labels,
        )
        loss = out.loss + self.mc_weight * out.mc_loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        mc = self._build_mc_batch(batch["input_ids"])

        lm_labels = self._mask_pads(mc["mc_input_ids"])
        out = self.model(
            input_ids      = mc["mc_input_ids"],
            attention_mask = mc["attention"],
            mc_token_ids   = mc["mc_token_ids"],
            mc_labels      = mc["mc_labels"],
            labels         = lm_labels,
        )
        loss = out.loss + self.mc_weight * out.mc_loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):

    #     gen_seq = self.model.generate(
    #                         input_ids=batch,
    #                         max_new_tokens=self.max_seq_length, 
    #                         do_sample=self.do_sample,
    #                         temperature=self.temperature,
    #                         top_k = self.top_k,
    #                         bos_token_id=self.start_token,
    #                         eos_token_id=self.end_token,
    #                         pad_token_id=self.pad_token,
    #                     )

    #     results = []
    #     gen_seq = gen_seq.detach().cpu()
        
    #     # preprocess results:
    #     for seq in gen_seq:
    #         seq = seq[1:] # rm start token
    #         seq[seq == self.end_token] = -1 # replace end token with -1
    #         seq[seq == self.pad_token] = -1 # replace pad token with -1

    #         if seq.numel() < self.max_seq_length:
    #             seq = F.pad(seq, (0, self.max_seq_length - seq.numel()), value=-1)
    #         else:
    #             seq = seq[: self.max_seq_length]
                
    #         results.append(seq)
        
    #     return torch.stack(results) #[:, :self.max_seq_length]


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Same contract as your old version:
        - input  : (B, L) tensor whose non‑pad part is just BOS + pads
        - output : (B, max_seq_length) constituent tokens, pads → -1
        """
        device   = batch.device
        results  = []

        for prompt in batch:
            prompt = prompt[prompt != self.pad_token].unsqueeze(0)  # strip pads
            gen    = self._generate_autostop(prompt, device=device) # 1‑D tensor

            seq = gen[1:]                     # drop BOS
            seq[seq == self.end_token] = -1
            seq[seq == self.pad_token] = -1

            if seq.numel() < self.max_seq_length:
                seq = F.pad(seq,
                            (0, self.max_seq_length - seq.numel()),
                            value=-1)
            else:
                seq = seq[: self.max_seq_length]

            results.append(seq)

        return torch.stack(results).detach().cpu() # (B, max_seq_length)

    #...helpers 

    def _mask_pads(self, x: torch.LongTensor) -> torch.LongTensor:
        """Replace pads with ‑100 so CE ignores them.
        """
        y = x.clone()
        y[y == self.pad_token] = -100
        return y

    def _build_mc_batch(self, input_ids: torch.LongTensor):
        """ Build MC tensors on the fly
        input_ids : (B, L)   full jets + BOS/EOS + pads
        returns    : dict with
           mc_input_ids  (B*2, S)
           mc_token_ids  (B, 2)
           mc_labels     (B,)        0 = stop gen, 1 = continue gen 
           attention     (B*2, S)
        """
        B, L_in = input_ids.shape
        device  = input_ids.device

        stop_seqs, cont_seqs, mc_labels = [], [], []
        cls_positions = []

        for seq in input_ids:          # loop over batch, cheap (<= 256)
            seq_trim = seq[seq != self.pad_token]
            total_len = seq_trim.size(0)             # incl. BOS & EOS
            # choose random prefix length  t  in  [1, total_len‑1]  (at least BOS)
            t = torch.randint(1, total_len, (1,)).item()

            prefix = seq_trim[:t]                    # (≤ total_len‑1,)
            stop_seq = torch.cat([prefix,
                                   torch.tensor([self.cls_token], device=device)])

            if t < total_len:    # prefix lacks end token → must continue
                continue_seq = torch.cat([prefix,
                                           seq_trim[t:t+1],            # next truth
                                           torch.tensor([self.cls_token], device=device)])
                mc_labels.append(1)                 # correct answer = continue
            else:                # prefix already complete
                continue_seq = stop_seq.clone()     # dummy
                mc_labels.append(0)

            stop_seqs.append(stop_seq)
            cont_seqs.append(continue_seq)
            cls_positions.append([stop_seq.size(0) - 1,
                                   continue_seq.size(0) - 1])

        # pad to common length S
        S = max(max(len(s) for s in stop_seqs),
                max(len(s) for s in cont_seqs))

        def pad_to(seq_list):
            out = torch.full((len(seq_list), S),
                             self.pad_token,
                             dtype=torch.long,
                             device=device)
            for i, s in enumerate(seq_list):
                out[i, :len(s)] = s
            return out

        # pad two choice lists to the same length S 
        stop_batch = pad_to(stop_seqs)      # (B, S)
        cont_batch = pad_to(cont_seqs)      # (B, S)
        mc_input_ids = torch.stack([stop_batch, cont_batch], dim=1)   # (B, 2, S)

        attention    = (mc_input_ids != self.pad_token).long()        # (B, 2, S)
        mc_token_ids = torch.tensor(cls_positions, device=device)     # (B, 2)
        mc_labels    = torch.tensor(mc_labels,    device=device)      # (B,)

        return {
            "mc_input_ids": mc_input_ids,   # (B,2,S)  
            "attention":    attention,      # (B,2,S)
            "mc_token_ids": mc_token_ids,   # (B,2)
            "mc_labels":    mc_labels,      # (B,)
        }

    #...autoregressive sampling

    @torch.no_grad()
    def _generate_autostop(self, prompt: torch.LongTensor, device: str = "cuda"):
        """
        Generate a jet sequence using:
        • LM head for token sampling,
        • MC head (stop vs continue) for learned termination.

        Stops when MC prob(stop) > θ *and* we have ≥ min_const constituents,
        or when LM emits <EOS>, or when max_seq_length reached.
        """
        gen  = prompt.to(device)          # (1, 1)  BOS
        past = None

        min_const  = 3                         # constituents after BOS
        stop_theta = max(self.stop_threshold, 0.85)

        specials = torch.tensor(
            [self.start_token, self.pad_token, self.cls_token], device=device
        )

        for _ in range(self.max_seq_length):
            # ─── 1) LM step ────────────────────────────────────────────────────
            lm_out = self.model(
                input_ids       = gen[:, -1:],
                past_key_values = past,
                use_cache       = True,
                return_dict     = True,
            )
            logits = lm_out.logits[:, -1, :].clone() / self.temperature
            logits[:, specials] = -float("inf")
            if gen.size(1) - 1 < min_const:
                logits[:, self.end_token] = -float("inf")
            if self.top_k:
                kth = torch.topk(logits, self.top_k).values[:, [-1]]
                logits[logits < kth] = -float("inf")

            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)  # (1,1)

            # accept token
            gen  = torch.cat([gen, next_tok], dim=-1)
            past = lm_out.past_key_values

            # immediate stop if EOS (and min length satisfied)
            if next_tok.item() == self.end_token and gen.size(1) - 1 >= min_const:
                break

            # skip MC check until min length
            if gen.size(1) - 1 < min_const:
                continue

            # ─── 2) Build stop / continue choices *with CLS last* ──────────────
            prefix_stop = gen.squeeze(0)                      # current prefix
            prefix_cont = torch.cat(
                [gen.squeeze(0),
                torch.argmax(logits, dim=-1).squeeze(0).unsqueeze(0)]  # deterministic alt
            )

            # pad shorter prefix so lengths equal
            S_pref = max(prefix_stop.size(0), prefix_cont.size(0))
            pad_len_stop = S_pref - prefix_stop.size(0)
            pad_len_cont = S_pref - prefix_cont.size(0)

            if pad_len_stop:
                prefix_stop = F.pad(prefix_stop, (0, pad_len_stop), value=self.pad_token)
            if pad_len_cont:
                prefix_cont = F.pad(prefix_cont, (0, pad_len_cont), value=self.pad_token)

            # now append CLS → CLS is at the *last* index = S_pref
            stop_seq = torch.cat([prefix_stop, torch.tensor([self.cls_token], device=device)])
            cont_seq = torch.cat([prefix_cont, torch.tensor([self.cls_token], device=device)])
            S = stop_seq.size(0)  # = cont_seq.size(0) = S_pref + 1

            mc_input_ids = torch.stack([stop_seq, cont_seq], dim=0).unsqueeze(0)  # (1,2,S)
            mc_token_ids = torch.tensor([[S - 1, S - 1]], device=device)          # CLS idx
            attention    = (mc_input_ids != self.pad_token).long()

            mc_logits = self.model(
                input_ids      = mc_input_ids,
                mc_token_ids   = mc_token_ids,
                attention_mask = attention,
                return_dict    = True,
            ).mc_logits                                            # (1,2)

            p_stop = torch.softmax(mc_logits, dim=-1)[0, 0].item()

            # ─── 3) stop / continue decision ──────────────────────────────────
            if p_stop > stop_theta:
                break

        return gen.squeeze(0)      # 1‑D LongTensor




    #...log‑prob utilities

    @torch.no_grad()
    def _log_probs(self, batch_ids, batch_mask, preprocessed=False):
        labels = batch_ids.clone()
        if preprocessed:
            labels[labels == -1] = -100
        else:
            labels[(labels == self.start_token) |
                   (labels == self.end_token)  |
                   (labels == self.pad_token)] = -100

        out = self.model(input_ids=batch_ids,
                         attention_mask=batch_mask,
                         labels=labels)
        logits = out.logits[:, :-1, :]         # drop pred for last token
        labels = labels[:, 1:]                 # shift
        logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                labels.reshape(-1),
                                reduction="none",
                                ignore_index=-100)
        return logp.view(batch_ids.size(0), -1).sum(1)  # (B,)

    @torch.no_grad()
    def log_probs(self, sample, batch_size=256, device="cuda"):
        self.model.eval().to(device)
        seq = sample.to(device)
        attn_mask = (seq != -1).long()
        dl = DataLoader(TensorDataset(seq, attn_mask), batch_size=batch_size)
        outs = [self._log_probs(b0, b1, preprocessed=True).cpu() for b0, b1 in dl]
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def per_token_preds(self, seq, device=None):
        self.model.eval().to(device)
        seq = torch.cat([torch.tensor([self.start_token], device=device), seq.to(device)])
        logits = self.model(input_ids=seq[:-1].unsqueeze(0)).logits
        log_probs = F.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(2, seq[1:].unsqueeze(0).unsqueeze(-1)).squeeze()
        preds = logits.argmax(-1).squeeze()
        return preds.cpu(), tok_lp.cpu()

    #...optimizer 
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        sch = CosineAnnealingLR(opt,
                                T_max=self.trainer.max_epochs,
                                eta_min=self.hparams.learning_rate_final)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch,
                                 "interval": "epoch"}}
