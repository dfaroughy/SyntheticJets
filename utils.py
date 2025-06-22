
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
import awkward as ak
import vector
import fastjet
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from pytorch_lightning.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from sklearn.metrics import roc_curve, auc

from datamodule_jetclass import JetSequence

vector.register_awkward()

class GeneratorCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experiment_dir = Path(f'{config.dir}/{config.project_name}/{config.experiment_id}')
        self.jet_type = config.jet_type
        self.data_dir = f'{config.dir}/JetClass' 
        self.tag = config.tag
        self.max_seq_length=config.max_seq_length

    def on_predict_start(self, trainer, pl_module):
        self.batched_data = []
        self.predict_type = trainer.model.predict_type

        self.num_bins = trainer.model.num_bins 
        self.logpt_range = trainer.model.logpt_range
        self.eta_range = trainer.model.eta_range
        self.phi_range = trainer.model.phi_range

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
        
        os.mkdir(f'{self.experiment_dir}/{self.predict_type}_results_{self.tag}')

        with open(f'{self.experiment_dir}/{self.predict_type}_results_{self.tag}/configs.yaml' , 'w' ) as outfile:
            yaml.dump( self.config.__dict__, outfile, sort_keys=False)

        temp_files = self.experiment_dir.glob(f"{self.predict_type}_temp_data_*_*.pt")
        data_tokens = torch.cat([torch.load(str(f)) for f in temp_files], dim=0)

        print(f'INFO: first event: {data_tokens[0]}')

        np.save(f'{self.experiment_dir}/{self.predict_type}_results_{self.tag}/{self.file_name}_tokens.npy', data_tokens)
        print(f'\nINFO: generated {data_tokens.shape[0]} jet sequences')
        print(f'INFO: data saved in {self.experiment_dir}/{self.predict_type}_results_{self.tag}')

        Jets = JetSequence(max_seq_length=self.max_seq_length,
                           num_bins=self.num_bins,
                           start_token=self.start_token,
                           end_token=self.end_token,
                           pad_token=self.pad_token,)

        data_tokens = torch.where(data_tokens>=self.start_token, -1 * torch.ones_like(data_tokens), data_tokens)
        data_binned = binnify(jets=Jets.seq_to_bins_decoding(data_tokens[:, 1:]), 
                              logpt_range=self.logpt_range, 
                              eta_range=self.eta_range, 
                              phi_range=self.phi_range, 
                              num_bins=self.num_bins, 
                              make_continuous=self.config.make_continuous
                              )

        np.save(f'{self.experiment_dir}/{self.predict_type}_results_{self.tag}/{self.file_name}_binned.npy', data_binned)
        print(f'INFO: saved binned jets with shape {data_binned.shape}')

        if self.config.plots:
            print('INFO: plotting results...')
            self._plot_results(data_binned, 
                            path=f'{self.experiment_dir}/{self.predict_type}_results_{self.tag}', 
                            N=100_000
                            )

    def _clean_temp_files(self):
        for f in self.experiment_dir.glob(f"{self.predict_type}_temp_data_*_*.pt"):
            f.unlink()

    def _plot_results(self, gen_binned, path, N=100_000):

        #...preprocess BOS/EOS
        gen = gen_binned[:N].clone()  # remove first/last tokens

        #...get test and Aachen data for comparison:

        test_seq = JetSequence(filepath=f'{self.data_dir}/test_20M_binned/test_{self.jet_type}_2M_bins403030.h5', 
                               max_seq_length=self.max_seq_length,
                               num_bins=self.num_bins,
                               start_token=self.start_token,
                               end_token=self.end_token,
                               pad_token=self.pad_token,)

        # test_seq = JetSequence(filepath=f'{self.data_dir}/train_12M_EPiC_FM_binned/{self.jet_type}_EPiC_FM_bins403030.h5', 
        #                        max_seq_length=self.max_seq_length,
        #                        num_jets_min=10_500_000,
        #                        num_jets=11_500_000,  
        #                        num_bins=self.num_bins,
        #                        start_token=self.start_token,
        #                        end_token=self.end_token,
        #                        pad_token=self.pad_token,
        #                        )

        test_disc = binnify(jets=torch.tensor(test_seq.data[:N]).long(),
                            logpt_range=self.logpt_range, 
                            eta_range=self.eta_range, 
                            phi_range=self.phi_range, 
                            num_bins=self.num_bins, 
                            make_continuous=self.config.make_continuous
                            )

        # aachen_seq = JetSequence(filepath=f'{self.data_dir}/{self.jet_type}_samples_samples_nsamples2000000_trunc_5000_0.h5', 
        #                          max_seq_length=self.max_seq_length,
        #                          num_bins=self.num_bins,
        #                          start_token=self.start_token,
        #                          end_token=self.end_token,
        #                          pad_token=self.pad_token,)

        # aachen = binnify(jets=torch.tensor(aachen_seq.data[:N]).long(), 
        #                  logpt_range=self.logpt_range, 
        #                  eta_range=self.eta_range, 
        #                  phi_range=self.phi_range, 
        #                  num_bins=self.num_bins, 
        #                  make_continuous=self.config.make_continuous
        #                  )

        #...plot:

        plot_kin_with_ratio(test_disc,
                           gen, 
                        #    aachen, 
                           path=path + '/particle_level_obs.png', 
                           jet=f'{self.jet_type}')

        plot_hl_with_ratio(test_disc, 
                           gen, 
                        #    aachen, 
                           path=path + '/jet_level_obs.png',
                           jet=f'{self.jet_type}')


class LogProbsCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.jet_type = config.jet_type
        self.eval_type = config.eval_data_type
        self.experiment_dir = Path(f'{config.dir}/{config.project_name}/{config.experiment_id}')
        self.predict_type = self.config.predict_type
        self.file_name = f'{self.jet_type}_{self.predict_type}_{self.eval_type}_{config.tag}'

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
        path = f"{self.experiment_dir}/{self.jet_type}_{self.predict_type}_{self.eval_type}_temp_rank_{rank}.pt"
        torch.save(data, path)


    @rank_zero_only
    def _gather_results_global(self, trainer):
        temp_files = sorted(self.experiment_dir.glob(f"{self.jet_type}_{self.predict_type}_{self.eval_type}_temp_rank_*.pt"))
        logprobs = torch.cat([torch.load(str(f)) for f in temp_files], dim=0)
        np.save(f'{self.experiment_dir}/{self.file_name}.npy', logprobs)
        print(f'\nINFO: computing log-likelihood: {self.file_name}')


    def _clean_temp_files(self):
        for f in self.experiment_dir.glob(f"{self.jet_type}_{self.predict_type}_{self.eval_type}_temp_rank_*.pt"):
            f.unlink()

    def _plot_results(self, logprobs, path, N=10_000):
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.histplot(logprobs[:N], bins=100,lw=0.4, fill=True, alpha=0.5,  color='k',  element='step', stat='density', ax=ax)
        ax.set_xlabel(r'log-prob')
        ax.set_xlabel(r'density')
        plt.tight_layout()
        plt.savefig(path, dpi=500, bbox_inches='tight')

class JetSubstructure:
    def __init__(self, constituents, R=0.8, beta=1.0, use_wta_scheme=True):

        pt = constituents[...,0] 
        eta = constituents[...,1]
        phi = constituents[...,2]

        constituents_ak = ak.zip(
            {
                "pt": np.array(pt),
                "eta": np.array(eta),
                "phi": np.array(phi),
                "mass": np.zeros_like(np.array(pt)),
            },
            with_name="Momentum4D",
        )

        constituents_ak = ak.mask(constituents_ak, constituents_ak.pt > 0)
        constituents_ak = ak.drop_none(constituents_ak)

        self._constituents_ak = constituents_ak[ak.num(constituents_ak) >= 3]

        if use_wta_scheme:
            jetdef = fastjet.JetDefinition(
                fastjet.kt_algorithm, R, fastjet.WTA_pt_scheme
            )
        else:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, R)

        print("Clustering jets with fastjet")
        print("Jet definition:", jetdef)
        print("Calculating N-subjettiness")

        self._cluster = fastjet.ClusterSequence(self._constituents_ak, jetdef)
        self.d0 = self._calc_d0(R, beta)
        self.c1 = self._cluster.exclusive_jets_energy_correlator(njets=1, func="c1")
        self.d2 = self._cluster.exclusive_jets_energy_correlator(njets=1, func="d2")
        self.tau1 = self._calc_tau1(beta)
        self.tau2 = self._calc_tau2(beta)
        self.tau3 = self._calc_tau3(beta)
        self.tau21 = np.ma.divide(self.tau2, self.tau1)
        self.tau32 = np.ma.divide(self.tau3, self.tau2)

    def _calc_deltaR(self, particles, jet):
        jet = ak.unflatten(ak.flatten(jet), counts=1)
        return particles.deltaR(jet)

    def _calc_d0(self, R, beta):
        """Calculate the d0 values."""
        return ak.sum(self._constituents_ak.pt * R**beta, axis=1)

    def _calc_tau1(self, beta):
        """Calculate the tau1 values."""
        excl_jets_1 = self._cluster.exclusive_jets(n_jets=1)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_1[:, :1])
        pt_i = self._constituents_ak.pt
        return ak.sum(pt_i * delta_r_1i**beta, axis=1) / self.d0

    def _calc_tau2(self, beta):
        """Calculate the tau2 values."""
        excl_jets_2 = self._cluster.exclusive_jets(n_jets=2)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_2[:, :1])
        delta_r_2i = self._calc_deltaR(self._constituents_ak, excl_jets_2[:, 1:2])
        pt_i = self._constituents_ak.pt

        #...add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** beta,
                    delta_r_2i[..., np.newaxis] ** beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        return ak.sum(pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self, beta):
        """Calculate the tau3 values."""
        excl_jets_3 = self._cluster.exclusive_jets(n_jets=3)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, :1])
        delta_r_2i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, 1:2])
        delta_r_3i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, 2:3])
        pt_i = self._constituents_ak.pt

        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** beta,
                    delta_r_2i[..., np.newaxis] ** beta,
                    delta_r_3i[..., np.newaxis] ** beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        return ak.sum(pt_i * min_delta_r, axis=1) / self.d0



def binnify(jets, 
            logpt_range=(-0.7602971186041831, 6.906254768371582),
            eta_range=(-0.8, 0.8),
            phi_range=(-0.8, 0.8), 
            num_bins: list = [40, 30, 30],
            make_continuous=False):

    """
    Take digitized log(pt), eta, phi bins data and maps to physical space

    log(pt) bins:  
    
    [-0.76029712 -0.56371887 -0.36714061 -0.17056236  0.0260159   0.22259415
    0.4191724   0.61575066  0.81232891  1.00890716  1.20548542  1.40206367
    1.59864192  1.79522018  1.99179843  2.18837668  2.38495494  2.58153319
    2.77811144  2.9746897   3.17126795  3.36784621  3.56442446  3.76100271
    3.95758097  4.15415922  4.35073747  4.54731573  4.74389398  4.94047223
    5.13705049  5.33362874  5.53020699  5.72678525  5.9233635   6.11994175
    6.31652001  6.51309826  6.70967651  6.90625477]

    eta bins:  
    
    [-0.8        -0.74482759 -0.68965517 -0.63448276 -0.57931034 -0.52413793
    -0.46896552 -0.4137931  -0.35862069 -0.30344828 -0.24827586 -0.19310345
    -0.13793103 -0.08275862 -0.02758621  0.02758621  0.08275862  0.13793103
    0.19310345  0.24827586  0.30344828  0.35862069  0.4137931   0.46896552
    0.52413793  0.57931034  0.63448276  0.68965517  0.74482759  0.8       ]

    phi bins:  
    
    [-0.8        -0.74482759 -0.68965517 -0.63448276 -0.57931034 -0.52413793
    -0.46896552 -0.4137931  -0.35862069 -0.30344828 -0.24827586 -0.19310345
    -0.13793103 -0.08275862 -0.02758621  0.02758621  0.08275862  0.13793103
    0.19310345  0.24827586  0.30344828  0.35862069  0.4137931   0.46896552
    0.52413793  0.57931034  0.63448276  0.68965517  0.74482759  0.8       ]

    """

    # pt_bins = np.load(f"{bin_dir}/preprocessing_bins/pt_bins_1Mfromeach_403030.npy")
    # eta_bins = np.load(f"{bin_dir}/preprocessing_bins/eta_bins_1Mfromeach_403030.npy")
    # phi_bins = np.load(f"{bin_dir}/preprocessing_bins/phi_bins_1Mfromeach_403030.npy")


    pt_bins = np.linspace(logpt_range[0], logpt_range[1], num_bins[0])
    eta_bins = np.linspace(eta_range[0], eta_range[1], num_bins[1])
    phi_bins = np.linspace(phi_range[0], phi_range[1], num_bins[2])

    pt_disc = jets[:, :, 0]
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    # rm pads or eos/bos tokens
    mask1 = pt_disc > len(pt_bins) 
    mask2 = pt_disc < 0

    d_pt = np.abs(pt_bins[1] - pt_bins[0])
    d_eta = np.abs(eta_bins[1] - eta_bins[0])
    d_phi = np.abs(phi_bins[1] - phi_bins[0])

    if make_continuous:
        pt_con = (pt_disc + np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * d_pt + pt_bins[0]
        eta_con = (eta_disc + np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * d_eta + eta_bins[0]
        phi_con = (phi_disc + np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * d_phi + phi_bins[0]
    
    else:
        pt_con = (pt_disc + 0.5) * d_pt + pt_bins[0]
        eta_con = (eta_disc + 0.5) * d_eta + eta_bins[0]
        phi_con = (phi_disc + 0.5) * d_phi + phi_bins[0]

    jets = np.stack((np.exp(pt_con), eta_con, phi_con), -1)
    jets[mask1] = 0
    jets[mask2] = 0
    jets = jets[:, 1:-1, :]  # remove start and end tokens

    return torch.tensor(jets)


# def make_continuous(jets, bin_dir='/pscratch/sd/d/dfarough/JetClass'):
#     pt_bins = np.load(f"{bin_dir}/preprocessing_bins/pt_bins_1Mfromeach_403030.npy")
#     eta_bins = np.load(f"{bin_dir}/preprocessing_bins/eta_bins_1Mfromeach_403030.npy")
#     phi_bins = np.load(f"{bin_dir}/preprocessing_bins/phi_bins_1Mfromeach_403030.npy")

#     pt_disc = jets[:, :, 0]
#     eta_disc = jets[:, :, 1]
#     phi_disc = jets[:, :, 2]

#     mask = pt_disc < 0

#     pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
#         pt_bins[1] - pt_bins[0]
#     ) + pt_bins[0]

#     eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
#         eta_bins[1] - eta_bins[0]
#     ) + eta_bins[0]
    
#     phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
#         phi_bins[1] - phi_bins[0]
#     ) + phi_bins[0]

#     continues_jets = np.stack((np.exp(pt_con), eta_con, phi_con), -1)
#     continues_jets[mask] = 0

#     return torch.tensor(continues_jets)


def jets_HighLevelFeats(sample):
    pt = sample[..., 0]
    eta = sample[..., 1]
    phi = sample[..., 2]

    # px,py,pz:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = pt * np.cosh(eta)

    jet_4mom = np.stack([px, py, pz, E], axis=-1)
    jet_4mom = jet_4mom.sum(axis=1)

    # Calculate the invariant mass of the jets
    jet_pt = np.sqrt(jet_4mom[..., 0]**2 + jet_4mom[..., 1]**2)
    jet_eta = np.arcsinh(jet_4mom[..., 2] / jet_pt)
    jet_phi = np.arctan2(jet_4mom[..., 1], jet_4mom[..., 0])
    jet_mass = np.sqrt(np.maximum(0, jet_4mom[..., 3]**2 - (jet_4mom[..., 0]**2 + jet_4mom[..., 1]**2 + jet_4mom[..., 2]**2)))

    return np.stack([jet_pt, jet_eta, jet_phi, jet_mass], axis=-1)


def plot_kin_with_ratio(test_disc, gen, aachen=None, test_cont=None, path='results_plot.png', jet='jetclass'):

    if test_cont is not None:
        test_cont = test_cont.cpu().numpy()
    test_disc = test_disc.cpu().numpy()
    gen = gen.cpu().numpy()
    
    if aachen is not None:
        aachen = aachen.cpu().numpy()

    bins = [np.arange(-1, 7, 0.2), 
            np.arange(-1, 1, 0.05), 
            np.arange(-1, 1, 0.05), 
            np.arange(0, 128, 2)]

    ylims = (0.4, 4.5, 4.5, 0.035)

    fig = plt.figure(figsize=(12, 2.5))
    gs = GridSpec(2, 4, height_ratios=(3,1), hspace=0.1, wspace=0.3)

    # --- TOP ROW: Hard‑coded histplots ---
    
    ax0 = fig.add_subplot(gs[0,0])
    if test_cont is not None:
        sns.histplot(np.log(test_cont[test_cont[...,0]>0][...,0]),bins=bins[0],lw=0.4, fill=False, ls=':',  color='k',  label=jet, element='step', stat='density', ax=ax0)
    sns.histplot(np.log(test_disc[test_disc[...,0]>0][...,0]),bins=bins[0],lw=0.4,fill=True,  color='k', alpha=0.2,  label=jet, element='step', stat='density', ax=ax0)
    sns.histplot(np.log(gen[gen[...,0]>0][...,0]), bins=bins[0], lw=0.8, fill=False, color='crimson', label='GPT2 Rutgers',element='step', stat='density', ax=ax0)
    if aachen is not None:
        sns.histplot(np.log(aachen[aachen[...,0]>0][...,0]), bins=bins[0], lw=0.8, fill=False, label='GPT2 Aachen',element='step', stat='density', ax=ax0)
    ax0.set_ylabel('density', fontsize=12)
    ax0.legend(fontsize=6)
    ax0.set_ylim(0, ylims[0])

    ax1 = fig.add_subplot(gs[0,1])
    if test_cont is not None:
        sns.histplot(test_cont[test_cont[...,0]>0][...,1], bins=bins[1], lw=0.4, fill=False, ls=':',  color='k',  element='step', stat='density', ax=ax1)
    sns.histplot(test_disc[test_disc[...,0]>0][...,1], bins=bins[1], lw=0.4, fill=True,  color='k', alpha=0.2,  element='step', stat='density', ax=ax1)
    sns.histplot(gen[gen[...,0]>0][...,1],  bins=bins[1], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax1)
    if aachen is not None:
        sns.histplot(aachen[aachen[...,0]>0][...,1], bins=bins[1], lw=0.8, fill=False, element='step', stat='density', ax=ax1)
    ax1.set_ylabel(' ', fontsize=12)
    ax1.set_ylim(0, ylims[1])

    ax2 = fig.add_subplot(gs[0,2])
    if test_cont is not None:
        sns.histplot(test_cont[test_cont[...,0]>0][...,2], bins=bins[2], lw=0.4, fill=False,  ls=':', color='k', element='step', stat='density', ax=ax2)    
    sns.histplot(test_disc[test_disc[...,0]>0][...,2], bins=bins[2], lw=0.4, fill=True,  color='k',  alpha=0.2, element='step', stat='density', ax=ax2)
    sns.histplot(gen[gen[...,0]>0][...,2],  bins=bins[2], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax2)
    if aachen is not None:
        sns.histplot(aachen[aachen[...,0]>0][...,2], bins=bins[2], lw=0.8, fill=False,element='step', stat='density', ax=ax2)
    ax2.set_ylabel(' ', fontsize=12)
    ax2.set_ylim(0, ylims[2])

    ax3 = fig.add_subplot(gs[0,3])
    sns.histplot((test_disc[...,0] > 0).sum(axis=1),  bins=bins[3], lw=0.4, fill=True,  color='k', alpha=0.2,element='step', stat='density', ax=ax3)
    sns.histplot((gen[...,0] > 0).sum(axis=1),  bins=bins[3], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax3)
    if aachen is not None:
        sns.histplot((aachen[...,0] > 0).sum(axis=1),  bins=bins[3], lw=0.8, fill=False, label='aachen',element='step', stat='density', ax=ax3)
    ax3.set_ylabel(' ', fontsize=12)
    ax3.set_ylim(0, ylims[3])

    # --- BOTTOM ROW: Ratio panels ---

    h0_t, e0 = np.histogram(np.log(test_disc[test_disc[...,0]>0][...,0]), bins=bins[0],      density=True)
    h0_g, _  = np.histogram(np.log(gen[gen[...,0]>0][...,0]),  bins=e0,         density=True)
    if aachen is not None:
        h0_a, _  = np.histogram(np.log(aachen[aachen[...,0]>0][...,0]), bins=e0,       density=True)
    centers0 = 0.5*(e0[:-1] + e0[1:])
    ax0r = fig.add_subplot(gs[1,0], sharex=ax0)
    ax0r.step(centers0, h0_g/(h0_t+1e-8), where='mid', color='crimson', lw=1)
    if aachen is not None:
        ax0r.step(centers0, h0_a/(h0_t+1e-8), where='mid',  lw=1)
    ax0r.set_ylim(0.7,1.3)
    ax0r.set_xlabel(r'$\log(p_T)$')
    ax0r.set_ylabel('ratio', fontsize=8)
    ax0r.axhline(y=1, color='k', linestyle='--', lw=0.75)

    h1_t, e1 = np.histogram(test_disc[test_disc[...,0]>0][...,1], bins=bins[1], density=True)
    h1_g, _  = np.histogram(gen[gen[...,0]>0][...,1],  bins=e1, density=True)
    if aachen is not None:
        h1_a, _  = np.histogram(aachen[aachen[...,0]>0][...,1], bins=e1, density=True)
    centers1 = 0.5*(e1[:-1] + e1[1:])
    ax1r = fig.add_subplot(gs[1,1], sharex=ax1)
    ax1r.step(centers1, h1_g/(h1_t+1e-8), where='mid',color='crimson',lw=1)
    if aachen is not None:
        ax1r.step(centers1, h1_a/(h1_t+1e-8), where='mid',   lw=1)
    ax1r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax1r.set_ylim(0.7,1.3)
    ax1r.set_xlabel(r'$\Delta\eta$')

    h2_t, e2 = np.histogram(test_disc[test_disc[...,0]>0][...,2], bins=bins[2],      density=True)
    h2_g, _  = np.histogram(gen[gen[...,0]>0][...,2],  bins=e2,         density=True)
    if aachen is not None:
        h2_a, _  = np.histogram(aachen[aachen[...,0]>0][...,2], bins=e2,       density=True)
    centers2 = 0.5*(e2[:-1] + e2[1:])
    ax2r = fig.add_subplot(gs[1,2], sharex=ax2)
    ax2r.step(centers2, h2_g/(h2_t+1e-8), where='mid', color='crimson',lw=1)
    if aachen is not None:
        ax2r.step(centers2, h2_a/(h2_t+1e-8), where='mid',  lw=1)
    ax2r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax2r.set_ylim(0.7,1.3)
    ax2r.set_xlabel(r'$\Delta\phi$')

    h3_t, e3 = np.histogram((test_disc[...,0] > 0).sum(axis=1), bins=bins[3], density=True)
    h3_g, _  = np.histogram((gen[...,0] > 0).sum(axis=1),  bins=e3,  density=True)
    if aachen is not None:
        h3_a, _  = np.histogram((aachen[...,0] > 0).sum(axis=1), bins=e3,density=True)
    centers3 = 0.5*(e3[:-1] + e3[1:])
    ax3r = fig.add_subplot(gs[1,3], sharex=ax3)
    ax3r.step(centers3, h3_g/(h3_t+1e-8), where='mid',color='crimson',lw=1)
    if aachen is not None:
        ax3r.step(centers3, h3_a/(h3_t+1e-8), where='mid',   lw=1)
    ax3r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax3r.set_ylim(0.7,1.3)
    ax3r.set_xlabel(r'$N$')

    plt.tight_layout()
    plt.savefig(path, dpi=500, bbox_inches='tight')


def plot_hl_with_ratio(test_disc, gen, aachen=None, test_cont=None, path='results_plot.png', jet='jetclass'):

    gen_HL = jets_HighLevelFeats(gen)
    if aachen is not None:
        aachen_HL = jets_HighLevelFeats(aachen)
    test_disc_HL = jets_HighLevelFeats(test_disc)

    gen_substructure = JetSubstructure(gen)
    if aachen is not None:
        aachen_substructure = JetSubstructure(aachen)
    test_disc_substructure = JetSubstructure(test_disc)

    if test_cont is not None:
        test_cont_HL = jets_HighLevelFeats(test_cont)
        test_cont_substructure = JetSubstructure(test_cont)    
    
    bins = [np.arange(400, 1200, 16), 
            np.arange(0., 500, 10), 
            np.arange(0, 1, 0.025), 
            np.arange(0, 1, 0.025)]

    ylims = ( 0.008 , 0.025, 5, 6)

    fig = plt.figure(figsize=(12, 2.5))
    gs = GridSpec(2, 4, height_ratios=(3,1), hspace=0.1, wspace=0.3)

    # --- TOP ROW: Hard‑coded histplots ---
    
    ax0 = fig.add_subplot(gs[0,0])
    if test_cont is not None:
        sns.histplot(test_cont_HL[...,0],bins=bins[0],lw=0.4,fill=False, ls=':', color='k',  label=jet, element='step', stat='density', ax=ax0)
    sns.histplot(test_disc_HL[...,0],bins=bins[0],lw=0.4,fill=True,  color='k', alpha=0.2,  label=jet + ' binned', element='step', stat='density', ax=ax0)
    sns.histplot(gen_HL[...,0], bins=bins[0], lw=0.8, fill=False, color='crimson', label='GPT2 Rutgers',element='step', stat='density', ax=ax0)
    if aachen is not None:
        sns.histplot(aachen_HL[...,0], bins=bins[0], lw=0.8, fill=False, label='GPT2 Aachen',element='step', stat='density', ax=ax0)
    ax0.set_ylabel('density', fontsize=12)
    ax0.set_ylim(0, ylims[0])
    ax0.legend(fontsize=7)

    ax1 = fig.add_subplot(gs[0,1])
    if test_cont is not None:
        sns.histplot(test_cont_HL[...,3], bins=bins[1], lw=0.4, fill=False, ls=':', color='k',  element='step', stat='density', ax=ax1)
    sns.histplot(test_disc_HL[...,3], bins=bins[1], lw=0.4, fill=True,  color='k', alpha=0.2,  element='step', stat='density', ax=ax1)
    sns.histplot(gen_HL[...,3],  bins=bins[1], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax1)
    if aachen is not None:
        sns.histplot(aachen_HL[...,3], bins=bins[1], lw=0.8, fill=False, element='step', stat='density', ax=ax1)
    ax1.set_ylabel(' ', fontsize=12)
    ax1.set_ylim(0, ylims[1])

    ax2 = fig.add_subplot(gs[0,2])
    if test_cont is not None:
        sns.histplot(test_cont_substructure.tau21, bins=bins[2], lw=0.4, fill=False,  color='k', ls=':', element='step', stat='density', ax=ax2)
    sns.histplot(test_disc_substructure.tau21, bins=bins[2], lw=0.4, fill=True,  color='k',  alpha=0.2, element='step', stat='density', ax=ax2)    
    sns.histplot(gen_substructure.tau21,  bins=bins[2], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax2)
    if aachen is not None:
        sns.histplot(aachen_substructure.tau21, bins=bins[2], lw=0.8, fill=False,element='step', stat='density', ax=ax2)
    ax2.set_ylabel(' ', fontsize=12)
    ax2.set_ylim(0, ylims[2])

    ax3 = fig.add_subplot(gs[0,3])
    if test_cont is not None:
        sns.histplot(test_cont_substructure.tau32, bins=bins[3], lw=0.4, fill=False,  color='k', ls=':', element='step', stat='density', ax=ax3)
    sns.histplot(test_disc_substructure.tau32, bins=bins[3], lw=0.4, fill=True,  color='k', alpha=0.2,element='step', stat='density', ax=ax3)
    sns.histplot(gen_substructure.tau32,  bins=bins[3], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax3)
    if aachen is not None:
        sns.histplot(aachen_substructure.tau32, bins=bins[3], lw=0.8, fill=False, label='aachen',element='step', stat='density', ax=ax3)
    ax3.set_ylabel(' ', fontsize=12)
    ax3.set_ylim(0, ylims[3])

    # --- BOTTOM ROW: Ratio panels ---

    h0_t, e0 = np.histogram(test_disc_HL[...,0], bins=bins[0],      density=True)
    h0_g, _  = np.histogram(gen_HL[...,0],  bins=e0,         density=True)
    if aachen is not None:
        h0_a, _  = np.histogram(aachen_HL[...,0], bins=e0,       density=True)
    centers0 = 0.5*(e0[:-1] + e0[1:])
    ax0r = fig.add_subplot(gs[1,0], sharex=ax0)
    ax0r.step(centers0, h0_g/(h0_t+1e-8), where='mid', color='crimson', lw=1)
    if aachen is not None:
        ax0r.step(centers0, h0_a/(h0_t+1e-8), where='mid',  lw=1)
    ax0r.set_ylim(0.7,1.3)
    ax0r.set_xlabel(r'jet $p_T$')
    ax0r.set_ylabel('ratio', fontsize=8)
    ax0r.axhline(y=1, color='k', linestyle='--', lw=0.75)

    h1_t, e1 = np.histogram(test_disc_HL[...,3], bins=bins[1], density=True)
    h1_g, _  = np.histogram(gen_HL[...,3],  bins=e1, density=True)
    if aachen is not None:
        h1_a, _  = np.histogram(aachen_HL[...,3], bins=e1, density=True)
    centers1 = 0.5*(e1[:-1] + e1[1:])
    ax1r = fig.add_subplot(gs[1,1], sharex=ax1)
    ax1r.step(centers1, h1_g/(h1_t+1e-8), where='mid',color='crimson',lw=1)
    if aachen is not None:
        ax1r.step(centers1, h1_a/(h1_t+1e-8), where='mid',   lw=1)
    ax1r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax1r.set_ylim(0.7,1.3)
    ax1r.set_xlabel(r'jet mass')

    h2_t, e2 = np.histogram(test_disc_substructure.tau21, bins=bins[2],      density=True)
    h2_g, _  = np.histogram(gen_substructure.tau21,  bins=e2,         density=True)
    if aachen is not None:
        h2_a, _  = np.histogram(aachen_substructure.tau21, bins=e2,       density=True)
    centers2 = 0.5*(e2[:-1] + e2[1:])
    ax2r = fig.add_subplot(gs[1,2], sharex=ax2)
    ax2r.step(centers2, h2_g/(h2_t+1e-8), where='mid', color='crimson',lw=1)
    if aachen is not None:
        ax2r.step(centers2, h2_a/(h2_t+1e-8), where='mid',  lw=1)
    ax2r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax2r.set_ylim(0.7,1.3)
    ax2r.set_xlabel(r'$\tau_{21}$')

    h3_t, e3 = np.histogram(test_disc_substructure.tau32, bins=bins[3],      density=True)
    h3_g, _  = np.histogram(gen_substructure.tau32,  bins=e3,         density=True)
    if aachen is not None:
        h3_a, _  = np.histogram(aachen_substructure.tau32, bins=e3,       density=True)
    centers3 = 0.5*(e3[:-1] + e3[1:])
    ax3r = fig.add_subplot(gs[1,3], sharex=ax3)
    ax3r.step(centers3, h3_g/(h3_t+1e-8), where='mid',color='crimson',lw=1)
    if aachen is not None:
        ax3r.step(centers3, h3_a/(h3_t+1e-8), where='mid',   lw=1)
    ax3r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax3r.set_ylim(0.7,1.3)
    ax3r.set_xlabel(r'$\tau_{32}$')

    plt.tight_layout()
    plt.savefig(path, dpi=500, bbox_inches='tight')


def ROC(LLR_bckg, LLR_sig, label):
    """
    Compute ROC curve and AUC
    """

    scores = np.concatenate([LLR_sig, LLR_bckg])
    labels = np.concatenate([np.ones(LLR_bckg.shape[0]), np.zeros(LLR_bckg.shape[0])])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(tpr, 1 / fpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("signal efficiency")
    plt.ylabel("background rejection")
    plt.title("ROC curve")
    plt.legend()
    plt.yscale("log")


# def kin_plots(toy_qcd, gen_jets, save_file='kin_plots.png'):

#     qcd_x = toy_qcd[:, :, 0].flatten()
#     qcd_y = toy_qcd[:, :, 1].flatten()
#     gen_x = gen_jets[:, :, 0].flatten()
#     gen_y = gen_jets[:, :, 1].flatten()

#     qcd_x_mean = toy_qcd[:, :, 0].mean(axis=1)
#     qcd_y_mean = toy_qcd[:, :, 1].mean(axis=1)
#     gen_x_mean = gen_jets[:, :, 0].mean(axis=1)
#     gen_y_mean = gen_jets[:, :, 1].mean(axis=1)

#     qcd_x_std = toy_qcd[:, :, 0].std(axis=1)
#     qcd_y_std = toy_qcd[:, :, 1].std(axis=1)
#     gen_x_std = gen_jets[:, :, 0].std(axis=1)
#     gen_y_std = gen_jets[:, :, 1].std(axis=1)


#     # --- Create subplots ---
#     fig, ax = plt.subplots(3, 2, figsize=(5, 5))

#     bins= np.linspace(0, 1, 11)

#     sns.histplot(qcd_x, bins=bins, fill=False, color='darkred', lw=1, label='synthetic data', element='step', stat='density', ax=ax[0, 0])
#     sns.histplot(gen_x, bins=bins, fill=False, color='crimson', lw=1, ls='--',  label='GPT-2', element='step', stat='density', ax=ax[0, 0])
#     ax[0,0].set_ylabel('Density')
#     ax[0,0].set_xlabel(r'$z$')
#     ax[0,0].set_yscale('log')
#     ax[0,0].legend(loc='upper right', fontsize=6)


#     bins=np.linspace(-15,15, 16)

#     sns.histplot(qcd_y, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[0, 1])
#     sns.histplot(gen_y, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[0, 1])
#     ax[0,1].set_ylabel('')
#     ax[0,1].set_xlabel(r'$\varphi$')

#     bins=np.linspace(0, 0.35, 20)

#     sns.histplot(qcd_x_mean, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[1, 0])
#     sns.histplot(gen_x_mean, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[1, 0])
#     ax[1,0].set_ylabel('Density')
#     ax[1,0].set_xlabel(r'$\langle z\rangle$')

#     bins=np.linspace(-6, 12, 30)

#     sns.histplot(qcd_y_mean, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[1, 1])
#     sns.histplot(gen_y_mean, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[1, 1])
#     ax[1,1].set_ylabel('')
#     ax[1,1].set_xlabel(r'$\langle\varphi\rangle$')

#     bins=np.linspace(0, 0.3, 30)

#     # Bottom row: standard deviation distributions
#     sns.histplot(qcd_x_std, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[2, 0])
#     sns.histplot(gen_x_std, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[2, 0])
#     ax[2,0].set_ylabel('Density')
#     ax[2,0].set_xlabel(r'$\sigma_z$')

#     bins=np.linspace(0, 7, 30)

#     sns.histplot(qcd_y_std, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[2, 1])
#     sns.histplot(gen_y_std, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[2, 1])
#     ax[2,1].set_ylabel('')
#     ax[2,1].set_xlabel(r'$\sigma_\varphi$')

#     custom_ticks = {
#         0: [0, 0.2, 0.4, 0.6, 0.8, 1],
#         1: [-10, -5, 0, 5, 10],
#         2: [0, 0.1, 0.2, 0.3, 0.4, 0.5] ,
#         3: [-10, -5, 0, 5, 10],
#         4: [0, 0.1, 0.2, 0.3, 0.4],
#         5: [1,2,3,4,5,6]
#     }

#     # For each axis, set custom ticks based on its column:
#     for i in range(3):  # rows
#         for j in range(2):  # columns
#             if j == 0:
#                 if i == 0:
#                     ax[i, j].set_xticks(custom_ticks[0])
#                 elif i == 1:
#                     ax[i, j].set_xticks(custom_ticks[2])
#                 elif i == 2:
#                     ax[i, j].set_xticks(custom_ticks[4])
#             elif j == 1:
#                 if i == 0:
#                     ax[i, j].set_xticks(custom_ticks[1])
#                 elif i == 1:
#                     ax[i, j].set_xticks(custom_ticks[3])
#                 elif i == 2:
#                     ax[i, j].set_xticks(custom_ticks[5])

#             ax[i, j].tick_params(axis='x', labelsize=8)
#             ax[i, j].tick_params(axis='y', labelsize=8)

#     ax[1,0].set_xlim(0, 0.5)
#     ax[0,1].set_xlim(-10, 12)
#     ax[1,1].set_xlim(-6, 12)
#     ax[2,0].set_xlim(0, 0.4)
#     ax[2,1].set_xlim(0., 7)

#     plt.tight_layout()
#     fig.subplots_adjust(hspace=0.65, wspace=0.3) 
#     plt.savefig(save_file, dpi=300, bbox_inches='tight')
#     plt.show()



# def ordered_z_plots(toy_qcd, gen_jets, save_file='ordered_z_plots.png', feat=0):

#     fig, ax = plt.subplots(3, 2, figsize=(6, 6))

#     bins=np.arange(0, 1, 0.01)

#     # --- Plot hardest ---

#     sns.histplot(toy_qcd[:,0,:][...,feat], bins=bins, alpha=0.25, color='darkred', lw=0, label='data', element='step', stat='density', ax=ax[0, 0])
#     sns.histplot(gen_jets[:,0,:][...,feat], bins=bins,  fill=False, color='crimson', lw=1, label='gpt2', element='step', stat='density', ax=ax[0, 0])

#     ax[0,0].set_ylabel('Density')
#     ax[0,0].set_xlabel(r'hardest $z$')
#     ax[0,0].legend(loc='upper left', fontsize=6)
#     ax[0,0].set_xlim(0, 1)


#     sns.histplot(toy_qcd[:,5,:][...,feat], bins=bins,   alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[0, 1])
#     sns.histplot(gen_jets[:,5,:][...,feat], bins=bins,  fill=False, color='crimson', lw=1, label='gpt2', element='step', stat='density', ax=ax[0, 1])
#     ax[0,1].set_ylabel('')
#     ax[0,1].set_xlabel(r'5th hardest $z$')


#     # bins=np.arange(0, 0.6, 0.01)

#     sns.histplot(toy_qcd[:,10,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[1, 0])
#     sns.histplot(gen_jets[:,10,:][...,feat], bins=bins,  fill=False, color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[1, 0])

#     ax[1,0].set_ylabel('Density')
#     ax[1,0].set_xlabel(r'10th hardest $z$')
#     # ax[1,0].legend(loc='upper right', fontsize=6)
#     # ax[1,0].set_xlim(0, 0.6)

#     # bins=np.arange(0, 0.6, 0.008)

#     sns.histplot(toy_qcd[:,15,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[1, 1])
#     sns.histplot(gen_jets[:,15,:][...,feat], bins=bins, fill=False, color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[1, 1])
#     ax[1,1].set_ylabel('')
#     ax[1,1].set_xlabel(r'15th hardest $z$')
#     # ax[1,1].legend(loc='upper right', fontsize=6)
#     # ax[1,1].set_xlim(0, 0.6)


#     # Middle row: intermediate distributions

#     # bins=np.arange(0, 0.2, 0.004)

#     sns.histplot(toy_qcd[:,20,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[2, 0])
#     sns.histplot(gen_jets[:,20,:][...,feat], bins=bins, fill=False,  color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[2, 0])

#     ax[2,0].set_ylabel('Density')
#     ax[2,0].set_xlabel(r' 20th hardest $z$')
#     # ax[2,0].legend(loc='upper right', fontsize=6)
#     # ax[2,0].set_xlim(0, 0.4)


#     # bins=np.arange(0, 0.05, 0.001)

#     sns.histplot(toy_qcd[:,25,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[2, 1])
#     sns.histplot(gen_jets[:,25,:][...,feat], bins=bins, fill=False,  color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[2, 1])
#     ax[2,1].set_ylabel('')
#     ax[2,1].set_xlabel(r'25th hardest $z$')
#     # ax[2,1].legend(loc='upper right', fontsize=6)
#     # ax[2,1].set_xlim(0, 0.1)

#     plt.tight_layout()
#     fig.subplots_adjust(hspace=0.65, wspace=0.3) 
#     plt.savefig(save_file, dpi=300, bbox_inches='tight')

#     plt.show()

