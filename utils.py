import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

import awkward as ak
import vector
import fastjet
from sklearn.metrics import roc_curve, auc
vector.register_awkward()



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


def kin_plots(toy_qcd, gen_jets, save_file='kin_plots.png'):

    qcd_x = toy_qcd[:, :, 0].flatten()
    qcd_y = toy_qcd[:, :, 1].flatten()
    gen_x = gen_jets[:, :, 0].flatten()
    gen_y = gen_jets[:, :, 1].flatten()

    qcd_x_mean = toy_qcd[:, :, 0].mean(axis=1)
    qcd_y_mean = toy_qcd[:, :, 1].mean(axis=1)
    gen_x_mean = gen_jets[:, :, 0].mean(axis=1)
    gen_y_mean = gen_jets[:, :, 1].mean(axis=1)

    qcd_x_std = toy_qcd[:, :, 0].std(axis=1)
    qcd_y_std = toy_qcd[:, :, 1].std(axis=1)
    gen_x_std = gen_jets[:, :, 0].std(axis=1)
    gen_y_std = gen_jets[:, :, 1].std(axis=1)


    # --- Create subplots ---
    fig, ax = plt.subplots(3, 2, figsize=(5, 5))

    bins= np.linspace(0, 1, 11)

    sns.histplot(qcd_x, bins=bins, fill=False, color='darkred', lw=1, label='synthetic data', element='step', stat='density', ax=ax[0, 0])
    sns.histplot(gen_x, bins=bins, fill=False, color='crimson', lw=1, ls='--',  label='GPT-2', element='step', stat='density', ax=ax[0, 0])
    ax[0,0].set_ylabel('Density')
    ax[0,0].set_xlabel(r'$z$')
    ax[0,0].set_yscale('log')
    ax[0,0].legend(loc='upper right', fontsize=6)


    bins=np.linspace(-15,15, 16)

    sns.histplot(qcd_y, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[0, 1])
    sns.histplot(gen_y, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[0, 1])
    ax[0,1].set_ylabel('')
    ax[0,1].set_xlabel(r'$\varphi$')

    bins=np.linspace(0, 0.35, 20)

    sns.histplot(qcd_x_mean, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[1, 0])
    sns.histplot(gen_x_mean, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[1, 0])
    ax[1,0].set_ylabel('Density')
    ax[1,0].set_xlabel(r'$\langle z\rangle$')

    bins=np.linspace(-6, 12, 30)

    sns.histplot(qcd_y_mean, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[1, 1])
    sns.histplot(gen_y_mean, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[1, 1])
    ax[1,1].set_ylabel('')
    ax[1,1].set_xlabel(r'$\langle\varphi\rangle$')

    bins=np.linspace(0, 0.3, 30)

    # Bottom row: standard deviation distributions
    sns.histplot(qcd_x_std, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[2, 0])
    sns.histplot(gen_x_std, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[2, 0])
    ax[2,0].set_ylabel('Density')
    ax[2,0].set_xlabel(r'$\sigma_z$')

    bins=np.linspace(0, 7, 30)

    sns.histplot(qcd_y_std, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[2, 1])
    sns.histplot(gen_y_std, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[2, 1])
    ax[2,1].set_ylabel('')
    ax[2,1].set_xlabel(r'$\sigma_\varphi$')

    custom_ticks = {
        0: [0, 0.2, 0.4, 0.6, 0.8, 1],
        1: [-10, -5, 0, 5, 10],
        2: [0, 0.1, 0.2, 0.3, 0.4, 0.5] ,
        3: [-10, -5, 0, 5, 10],
        4: [0, 0.1, 0.2, 0.3, 0.4],
        5: [1,2,3,4,5,6]
    }

    # For each axis, set custom ticks based on its column:
    for i in range(3):  # rows
        for j in range(2):  # columns
            if j == 0:
                if i == 0:
                    ax[i, j].set_xticks(custom_ticks[0])
                elif i == 1:
                    ax[i, j].set_xticks(custom_ticks[2])
                elif i == 2:
                    ax[i, j].set_xticks(custom_ticks[4])
            elif j == 1:
                if i == 0:
                    ax[i, j].set_xticks(custom_ticks[1])
                elif i == 1:
                    ax[i, j].set_xticks(custom_ticks[3])
                elif i == 2:
                    ax[i, j].set_xticks(custom_ticks[5])

            ax[i, j].tick_params(axis='x', labelsize=8)
            ax[i, j].tick_params(axis='y', labelsize=8)

    ax[1,0].set_xlim(0, 0.5)
    ax[0,1].set_xlim(-10, 12)
    ax[1,1].set_xlim(-6, 12)
    ax[2,0].set_xlim(0, 0.4)
    ax[2,1].set_xlim(0., 7)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.65, wspace=0.3) 
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()



def ordered_z_plots(toy_qcd, gen_jets, save_file='ordered_z_plots.png', feat=0):

    fig, ax = plt.subplots(3, 2, figsize=(6, 6))

    bins=np.arange(0, 1, 0.01)

    # --- Plot hardest ---

    sns.histplot(toy_qcd[:,0,:][...,feat], bins=bins, alpha=0.25, color='darkred', lw=0, label='data', element='step', stat='density', ax=ax[0, 0])
    sns.histplot(gen_jets[:,0,:][...,feat], bins=bins,  fill=False, color='crimson', lw=1, label='gpt2', element='step', stat='density', ax=ax[0, 0])

    ax[0,0].set_ylabel('Density')
    ax[0,0].set_xlabel(r'hardest $z$')
    ax[0,0].legend(loc='upper left', fontsize=6)
    ax[0,0].set_xlim(0, 1)


    sns.histplot(toy_qcd[:,5,:][...,feat], bins=bins,   alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[0, 1])
    sns.histplot(gen_jets[:,5,:][...,feat], bins=bins,  fill=False, color='crimson', lw=1, label='gpt2', element='step', stat='density', ax=ax[0, 1])
    ax[0,1].set_ylabel('')
    ax[0,1].set_xlabel(r'5th hardest $z$')


    # bins=np.arange(0, 0.6, 0.01)

    sns.histplot(toy_qcd[:,10,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[1, 0])
    sns.histplot(gen_jets[:,10,:][...,feat], bins=bins,  fill=False, color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[1, 0])

    ax[1,0].set_ylabel('Density')
    ax[1,0].set_xlabel(r'10th hardest $z$')
    # ax[1,0].legend(loc='upper right', fontsize=6)
    # ax[1,0].set_xlim(0, 0.6)

    # bins=np.arange(0, 0.6, 0.008)

    sns.histplot(toy_qcd[:,15,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[1, 1])
    sns.histplot(gen_jets[:,15,:][...,feat], bins=bins, fill=False, color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[1, 1])
    ax[1,1].set_ylabel('')
    ax[1,1].set_xlabel(r'15th hardest $z$')
    # ax[1,1].legend(loc='upper right', fontsize=6)
    # ax[1,1].set_xlim(0, 0.6)


    # Middle row: intermediate distributions

    # bins=np.arange(0, 0.2, 0.004)

    sns.histplot(toy_qcd[:,20,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[2, 0])
    sns.histplot(gen_jets[:,20,:][...,feat], bins=bins, fill=False,  color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[2, 0])

    ax[2,0].set_ylabel('Density')
    ax[2,0].set_xlabel(r' 20th hardest $z$')
    # ax[2,0].legend(loc='upper right', fontsize=6)
    # ax[2,0].set_xlim(0, 0.4)


    # bins=np.arange(0, 0.05, 0.001)

    sns.histplot(toy_qcd[:,25,:][...,feat], bins=bins,  alpha=0.25, color='darkred', lw=0, label='B', element='step', stat='density', ax=ax[2, 1])
    sns.histplot(gen_jets[:,25,:][...,feat], bins=bins, fill=False,  color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[2, 1])
    ax[2,1].set_ylabel('')
    ax[2,1].set_xlabel(r'25th hardest $z$')
    # ax[2,1].legend(loc='upper right', fontsize=6)
    # ax[2,1].set_xlim(0, 0.1)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.65, wspace=0.3) 
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    plt.show()



def make_continuous(jets):
    pt_bins = np.load("data/preprocessing_bins/pt_bins_1Mfromeach_403030.npy")
    eta_bins = np.load("data/preprocessing_bins/eta_bins_1Mfromeach_403030.npy")
    phi_bins = np.load("data/preprocessing_bins/phi_bins_1Mfromeach_403030.npy")

    pt_disc = jets[:, :, 0]
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    mask = pt_disc < 0

    # pt_con = pt_disc * (pt_bins[1] - pt_bins[0]) + pt_bins[0]
    # eta_con = eta_disc * (eta_bins[1] - eta_bins[0]) + eta_bins[0]
    # phi_con = phi_disc * (phi_bins[1] - phi_bins[0]) + phi_bins[0]

    pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
        pt_bins[1] - pt_bins[0]
    ) + pt_bins[0]

    eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
        eta_bins[1] - eta_bins[0]
    ) + eta_bins[0]
    
    phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
        phi_bins[1] - phi_bins[0]
    ) + phi_bins[0]

    continues_jets = np.stack((np.exp(pt_con), eta_con, phi_con), -1)
    continues_jets[mask] = 0

    return torch.tensor(continues_jets)


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

        # add new axis to make it broadcastable
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



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def plot_hl_with_ratio(test, gen, aachen, bins, ylims, jet='jetclass'):
    """
    Plot 4 seaborn histograms of jet high-level features (hard‑coded) with ratio panels below.

    Parameters
    ----------
    tops_HL_test, tops_HL_gen, tops_HL_aachen : np.ndarray
        Arrays of shape (N, 4), where columns are [pT, eta, phi, m].
    """

    fig = plt.figure(figsize=(12, 2.5))
    gs = GridSpec(2, 4, height_ratios=(3,1), hspace=0.1, wspace=0.3)

    # --- TOP ROW: Hard‑coded histplots ---
    
    ax0 = fig.add_subplot(gs[0,0])
    sns.histplot(test[...,0],bins=bins[0],lw=0.4,fill=True,  color='k', alpha=0.2,  label=jet, element='step', stat='density', ax=ax0)
    sns.histplot(gen[...,0], bins=bins[0], lw=0.8, fill=False, color='crimson', label='GPT2 Rutgers',element='step', stat='density', ax=ax0)
    sns.histplot(aachen[...,0], bins=bins[0], lw=0.8, fill=False, label='GPT2 Aachen',element='step', stat='density', ax=ax0)
    ax0.set_ylabel('density', fontsize=12)
    ax0.set_ylim(0, ylims[0])
    ax0.legend(fontsize=7)

    ax1 = fig.add_subplot(gs[0,1])
    sns.histplot(test[...,1], bins=bins[1], lw=0.4, fill=True,  color='k', alpha=0.2,  element='step', stat='density', ax=ax1)
    sns.histplot(gen[...,1],  bins=bins[1], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax1)
    sns.histplot(aachen[...,1], bins=bins[1], lw=0.8, fill=False, element='step', stat='density', ax=ax1)
    ax1.set_ylabel(' ', fontsize=12)
    ax1.set_ylim(0, ylims[1])

    ax2 = fig.add_subplot(gs[0,2])
    sns.histplot(test[...,2], bins=bins[2], lw=0.4, fill=True,  color='k',  alpha=0.2, element='step', stat='density', ax=ax2)
    sns.histplot(gen[...,2],  bins=bins[2], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax2)
    sns.histplot(aachen[...,2], bins=bins[2], lw=0.8, fill=False,element='step', stat='density', ax=ax2)
    ax2.set_ylabel(' ', fontsize=12)
    ax2.set_ylim(0, ylims[2])

    ax3 = fig.add_subplot(gs[0,3])
    sns.histplot(test[...,3], bins=bins[3], lw=0.4, fill=True,  color='k', alpha=0.2,element='step', stat='density', ax=ax3)
    sns.histplot(gen[...,3],  bins=bins[3], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax3)
    sns.histplot(aachen[...,3], bins=bins[3], lw=0.8, fill=False, label='aachen',element='step', stat='density', ax=ax3)
    ax3.set_ylabel(' ', fontsize=12)
    ax3.set_ylim(0, ylims[3])

    # --- BOTTOM ROW: Ratio panels ---

    h0_t, e0 = np.histogram(test[...,0], bins=bins[0],      density=True)
    h0_g, _  = np.histogram(gen[...,0],  bins=e0,         density=True)
    h0_a, _  = np.histogram(aachen[...,0], bins=e0,       density=True)
    centers0 = 0.5*(e0[:-1] + e0[1:])

    ax0r = fig.add_subplot(gs[1,0], sharex=ax0)
    ax0r.step(centers0, h0_g/(h0_t+1e-8), where='mid', color='crimson', lw=1)
    ax0r.step(centers0, h0_a/(h0_t+1e-8), where='mid',  lw=1)
    ax0r.set_ylim(0.5,1.5)
    ax0r.set_xlabel(r'jet $p_T$')
    ax0r.set_ylabel('ratio', fontsize=8)
    ax0r.axhline(y=1, color='k', linestyle='--', lw=0.75)

    h1_t, e1 = np.histogram(test[...,1], bins=bins[1],      density=True)
    h1_g, _  = np.histogram(gen[...,1],  bins=e1,         density=True)
    h1_a, _  = np.histogram(aachen[...,1], bins=e1,       density=True)
    centers1 = 0.5*(e1[:-1] + e1[1:])

    ax1r = fig.add_subplot(gs[1,1], sharex=ax1)
    ax1r.step(centers1, h1_g/(h1_t+1e-8), where='mid', color='crimson', lw=1)
    ax1r.step(centers1, h1_a/(h1_t+1e-8), where='mid', lw=1)
    ax1r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax1r.set_ylim(0.5,1.5)
    ax1r.set_xlabel(r'jet $\eta$')


    h2_t, e2 = np.histogram(test[...,2], bins=bins[2],      density=True)
    h2_g, _  = np.histogram(gen[...,2],  bins=e2,         density=True)
    h2_a, _  = np.histogram(aachen[...,2], bins=e2,       density=True)
    centers2 = 0.5*(e2[:-1] + e2[1:])

    ax2r = fig.add_subplot(gs[1,2], sharex=ax2)
    ax2r.step(centers2, h2_g/(h2_t+1e-8), where='mid', color='crimson',lw=1)
    ax2r.step(centers2, h2_a/(h2_t+1e-8), where='mid',  lw=1)
    ax2r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax2r.set_ylim(0.5,1.5)
    ax2r.set_xlabel(r'jet $\phi$')

    h3_t, e3 = np.histogram(test[...,3], bins=bins[3],      density=True)
    h3_g, _  = np.histogram(gen[...,3],  bins=e3,         density=True)
    h3_a, _  = np.histogram(aachen[...,3], bins=e3,       density=True)
    centers3 = 0.5*(e3[:-1] + e3[1:])

    ax3r = fig.add_subplot(gs[1,3], sharex=ax3)
    ax3r.step(centers3, h3_g/(h3_t+1e-8), where='mid',color='crimson',lw=1)
    ax3r.step(centers3, h3_a/(h3_t+1e-8), where='mid',   lw=1)
    ax3r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax3r.set_ylim(0.5,1.5)
    ax3r.set_xlabel(r'jet mass')

    plt.tight_layout()
    plt.show()



def plot_substructure_with_ratio(test, gen, aachen, bins, ylims, jet='jetclass' ):
    """
    Plot 4 seaborn histograms of jet high-level features (hard‑coded) with ratio panels below.

    Parameters
    ----------
    tops_HL_test, tops_HL_gen, tops_HL_aachen : np.ndarray
        Arrays of shape (N, 4), where columns are [pT, eta, phi, m].
    """

    fig = plt.figure(figsize=(12, 2.5))
    gs = GridSpec(2, 4, height_ratios=(3,1), hspace=0.1, wspace=0.3)

    # --- TOP ROW: Hard‑coded histplots ---
    ax0 = fig.add_subplot(gs[0,0])
    sns.histplot(test.c1, bins=bins[0],lw=0.4,fill=True,  color='k', alpha=0.2,  label=jet, element='step', stat='density', ax=ax0)
    sns.histplot(gen.c1, bins=bins[0], lw=0.8, fill=False, color='crimson', label='GPT2 Rutgers',element='step', stat='density', ax=ax0)
    sns.histplot(aachen.c1, bins=bins[0], lw=0.8, fill=False, label='GPT2 Aachen',element='step', stat='density', ax=ax0)
    ax0.set_ylabel('density', fontsize=12)
    ax0.set_ylim(0, ylims[0])
    ax0.legend(fontsize=7)

    ax1 = fig.add_subplot(gs[0,1])
    sns.histplot(test.d2, bins=bins[1], lw=0.4, fill=True,  color='k', alpha=0.2,  element='step', stat='density', ax=ax1)
    sns.histplot(gen.d2,  bins=bins[1], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax1)
    sns.histplot(aachen.d2, bins=bins[1], lw=0.8, fill=False, element='step', stat='density', ax=ax1)
    ax1.set_ylabel(' ', fontsize=12)
    ax1.set_ylim(0, ylims[1])

    ax2 = fig.add_subplot(gs[0,2])
    sns.histplot(test.tau21, bins=bins[2], lw=0.4, fill=True,  color='k',  alpha=0.2, element='step', stat='density', ax=ax2)
    sns.histplot(gen.tau21,  bins=bins[2], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax2)
    sns.histplot(aachen.tau21, bins=bins[2], lw=0.8, fill=False,element='step', stat='density', ax=ax2)
    ax2.set_ylabel(' ', fontsize=12)
    ax2.set_ylim(0, ylims[2])

    ax3 = fig.add_subplot(gs[0,3])
    sns.histplot(test.tau32, bins=bins[3], lw=0.4, fill=True,  color='k', alpha=0.2,element='step', stat='density', ax=ax3)
    sns.histplot(gen.tau32,  bins=bins[3], lw=0.8, fill=False, color='crimson', element='step', stat='density', ax=ax3)
    sns.histplot(aachen.tau32, bins=bins[3], lw=0.8, fill=False, label='aachen',element='step', stat='density', ax=ax3)
    ax3.set_ylabel(' ', fontsize=12)
    ax3.set_ylim(0, ylims[3])
    
    # --- BOTTOM ROW: Ratio panels ---

    h0_t, e0 = np.histogram(test.c1, bins=bins[0],      density=True)
    h0_g, _  = np.histogram(gen.c1,  bins=e0,         density=True)
    h0_a, _  = np.histogram(aachen.c1, bins=e0,       density=True)
    centers0 = 0.5*(e0[:-1] + e0[1:])

    ax0r = fig.add_subplot(gs[1,0], sharex=ax0)
    ax0r.step(centers0, h0_g/(h0_t+1e-8), where='mid', color='crimson', lw=1)
    ax0r.step(centers0, h0_a/(h0_t+1e-8), where='mid',  lw=1)
    ax0r.set_ylim(0.5,1.5)
    ax0r.set_xlabel(r'$C_1$')
    ax0r.set_ylabel('ratio', fontsize=8)
    ax0r.axhline(y=1, color='k', linestyle='--', lw=0.75)

    h1_t, e1 = np.histogram(test.d2, bins=bins[1],      density=True)
    h1_g, _  = np.histogram(gen.d2,  bins=e1,         density=True)
    h1_a, _  = np.histogram(aachen.d2, bins=e1,       density=True)
    centers1 = 0.5*(e1[:-1] + e1[1:])

    ax1r = fig.add_subplot(gs[1,1], sharex=ax1)
    ax1r.step(centers1, h1_g/(h1_t+1e-8), where='mid', color='crimson', lw=1)
    ax1r.step(centers1, h1_a/(h1_t+1e-8), where='mid', lw=1)
    ax1r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax1r.set_ylim(0.5,1.5)
    ax1r.set_xlabel(r'$D_2$')


    h2_t, e2 = np.histogram(test.tau21, bins=bins[2],      density=True)
    h2_g, _  = np.histogram(gen.tau21,  bins=e2,         density=True)
    h2_a, _  = np.histogram(aachen.tau21, bins=e2,       density=True)
    centers2 = 0.5*(e2[:-1] + e2[1:])

    ax2r = fig.add_subplot(gs[1,2], sharex=ax2)
    ax2r.step(centers2, h2_g/(h2_t+1e-8), where='mid', color='crimson',lw=1)
    ax2r.step(centers2, h2_a/(h2_t+1e-8), where='mid',  lw=1)
    ax2r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax2r.set_ylim(0.5,1.5)
    ax2r.set_xlabel(r'$\tau_{21}$')

    h3_t, e3 = np.histogram(test.tau32, bins=bins[3],      density=True)
    h3_g, _  = np.histogram(gen.tau32,  bins=e3,         density=True)
    h3_a, _  = np.histogram(aachen.tau32, bins=e3,       density=True)
    centers3 = 0.5*(e3[:-1] + e3[1:])

    ax3r = fig.add_subplot(gs[1,3], sharex=ax3)
    ax3r.step(centers3, h3_g/(h3_t+1e-8), where='mid',color='crimson',lw=1)
    ax3r.step(centers3, h3_a/(h3_t+1e-8), where='mid',   lw=1)
    ax3r.axhline(y=1, color='k', linestyle='--', lw=0.75)
    ax3r.set_ylim(0.5,1.5)
    ax3r.set_xlabel(r'$\tau_{32}$')

    plt.tight_layout()
    plt.show()
