import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from pytorch_lightning.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from sklearn.metrics import roc_curve, auc

from datamodule_jetclass import JetSequence
from utils import 

class GeneratorCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experiment_dir = Path(f'{config.dir}/{config.project_name}/{config.experiment_id}')
        self.jet_type = config.jet_type
        self.data_dir = f'{config.dir}/JetClass' 
        self.tag = config.tag

    def on_predict_start(self, trainer, pl_module):
        self.batched_data = []
        self.predict_type = trainer.model.predict_type
        self.start_token = trainer.model.start_token
        self.end_token = trainer.model.end_token
        self.pad_token = trainer.model.pad_token
        self.file_name = f'{self.predict_type}_{self.jet_type}_seq_{self.tag}'

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batched_data.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank

        self._save_results_local(rank)
        trainer.strategy.barrier()  # wait for all ranks to finish

        if trainer.is_global_zero:
            self._gather_results_global(trainer)
            self._clean_temp_files()

    def _save_results_local(self, rank):
        data = torch.cat(self.batched_data, dim=0)
        random = np.random.randint(0, 1000)
        path = f"{self.experiment_dir}/{self.predict_type}_temp_data_{rank}_{random}.pt"
        torch.save(data, path)

    @rank_zero_only
    def _gather_results_global(self, trainer):
        
        suffix = np.random.randint(0, 1000)
        os.mkdir(f'{self.experiment_dir}/{self.predict_type}_results_{suffix}_{self.tag}')

        with open(f'{self.experiment_dir}/{self.predict_type}_results_{suffix}_{self.tag}/configs.yaml' , 'w' ) as outfile:
            yaml.dump( self.config.__dict__, outfile, sort_keys=False)

        temp_files = self.experiment_dir.glob(f"{self.predict_type}_temp_data_*_*.pt")
        data_tokens = torch.cat([torch.load(str(f)) for f in temp_files], dim=0)

        print(f'INFO: first event: {data_tokens[0]}')

        np.save(f'{self.experiment_dir}/{self.predict_type}_results_{suffix}_{self.tag}/{self.file_name}_tokens.npy', data_tokens)
        print(f'\nINFO: generated {data_tokens.shape[0]} jet sequences')
        print(f'INFO: data saved in {self.experiment_dir}/{self.predict_type}_results_{suffix}_{self.tag}')

        Jets = JetSequence()

        data_tokens = torch.where(data_tokens>=self.start_token, -1 * torch.ones_like(data_tokens), data_tokens)
        data_binned = make_continuous(Jets.seq_to_bins_decoding(data_tokens[:, 1:]), self.data_dir) # rm start token

        np.save(f'{self.experiment_dir}/{self.predict_type}_results_{suffix}_{self.tag}/{self.file_name}_binned.npy', data_binned)
        print(f'INFO: saved binned jets with shape {data_binned.shape}')

        self._plot_results(data_binned, 
                           path=f'{self.experiment_dir}/{self.predict_type}_results_{suffix}_{self.tag}', 
                           N=1_000_000
                           )

    def _clean_temp_files(self):
        for f in self.experiment_dir.glob(f"{self.predict_type}_temp_data_*_*.pt"):
            f.unlink()

    def _plot_results(self, gen_binned, path, N=100_000):

        #...preprocess BOS/EOS
        gen = gen_binned[:N].clone()  # remove first/last tokens

        #...get test and Aachen data for comparison:

        test_seq = JetSequence(filepath=f'{self.data_dir}/test_20M_binned/test_{self.jet_type}_2M_bins403030.h5')
        test_disc = make_continuous(torch.tensor(test_seq.data[:N]).long(), self.data_dir)
        # test_cont = torch.tensor(test_seq.raw[:N]).long()

        aachen_seq = JetSequence(filepath=f'{self.data_dir}/{self.jet_type}_samples_samples_nsamples2000000_trunc_5000_0.h5',  )
        aachen = make_continuous(torch.tensor(aachen_seq.data[:N]).long(), self.data_dir)

        #...plot:

        plot_kin_with_ratio(test_disc, #test_cont,
                           gen, 
                           aachen, 
                           path=path + '/particle_level_obs.png', 
                           jet=f'{self.jet_type}')

        plot_hl_with_ratio(test_disc, #test_cont,
                           gen, 
                           aachen, 
                           path=path + '/jet_level_obs.png',
                           jet=f'{self.jet_type}')


class LogProbsCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_eval_file = Path(f'{config.dir}/{config.project_name}/{config.experiment_id}')
        self.jet_type = config.jet_type
        self.data_dir = f'{config.dir}/JetClass' 
        self.tag = config.tag

    def on_predict_start(self, trainer, pl_module):
        self.batched_data = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batched_data.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank

        self._save_results_local(rank)
        trainer.strategy.barrier()  # wait for all ranks to finish

        if trainer.is_global_zero:
            self._gather_results_global(trainer)
            self._clean_temp_files()

    def _save_results_local(self, rank):
        data = torch.cat(self.batched_data, dim=0)
        random = np.random.randint(0, 10000)
        path = f"{self.experiment_dir}/logp_temp_data_{rank}_{random}.pt"
        torch.save(data, path)

    @rank_zero_only
    def _gather_results_global(self, trainer):
        suffix = np.random.randint(0, 1000)
        os.mkdir(f'{self.experiment_dir}/logp_results_{suffix}_{self.tag}')

        with open(f'{self.experiment_dir}/logp_results_{suffix}_{self.tag}/configs.yaml' , 'w' ) as outfile:
            yaml.dump( self.config.__dict__, outfile, sort_keys=False)

        temp_files = self.experiment_dir.glob("logp_temp_data_*_*.pt")
        logprobs = torch.cat([torch.load(str(f)) for f in temp_files], dim=0)
        np.save(f'{self.experiment_dir}/logp_results_{suffix}_{self.tag}/{self.file_name}_tokens.npy', data_tokens)
        print(f'\nINFO: computing log-liklihood of provided jet sequences')
        print(f'INFO: logprobs saved in {self.experiment_dir}/logp_results_{suffix}_{self.tag}')

        self._plot_results(logprobs, path=f'{self.experiment_dir}/logp_results_{suffix}_{self.tag}', N=1_000_000)


    def _clean_temp_files(self):
        for f in self.experiment_dir.glob("logp_temp_data_*_*.pt"):
            f.unlink()

    def _plot_results(self, gen_binned, path, N=50_000):

        sns.histplot(logprobs)