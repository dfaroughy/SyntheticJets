import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

    bins= np.linspace(0, 1, 16)

    sns.histplot(qcd_x, bins=bins, fill=False, color='darkred', lw=1, label='synthetic data', element='step', stat='density', ax=ax[0, 0])
    sns.histplot(gen_x, bins=bins, fill=False, color='crimson', lw=1, ls='--',  label='GPT-2', element='step', stat='density', ax=ax[0, 0])
    ax[0,0].set_ylabel('Density')
    ax[0,0].set_xlabel(r'$z$')
    ax[0,0].set_yscale('log')
    ax[0,0].legend(loc='upper right', fontsize=6)


    bins=np.linspace(-10,10, 21)

    sns.histplot(qcd_y, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[0, 1])
    sns.histplot(gen_y, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[0, 1])
    ax[0,1].set_ylabel('')
    ax[0,1].set_xlabel(r'$\varphi$')

    bins=np.arange(0, 0.5, 0.02)

    sns.histplot(qcd_x_mean, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[1, 0])
    sns.histplot(gen_x_mean, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[1, 0])
    ax[1,0].set_ylabel('Density')
    ax[1,0].set_xlabel(r'$\langle z\rangle$')

    bins=np.arange(-5, 5, 0.25)

    sns.histplot(qcd_y_mean, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[1, 1])
    sns.histplot(gen_y_mean, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[1, 1])
    ax[1,1].set_ylabel('')
    ax[1,1].set_xlabel(r'$\langle\varphi\rangle$')

    bins=np.arange(0, 1, 0.01)

    # Bottom row: standard deviation distributions
    sns.histplot(qcd_x_std, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[2, 0])
    sns.histplot(gen_x_std, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[2, 0])
    ax[2,0].set_ylabel('Density')
    ax[2,0].set_xlabel(r'$\sigma_z$')

    bins=np.arange(0.5, 1.5, 0.03)

    sns.histplot(qcd_y_std, bins=bins, fill=False, color='darkred', lw=1, label='qcd', element='step', stat='density', ax=ax[2, 1])
    sns.histplot(gen_y_std, bins=bins, fill=False, color='crimson', lw=1, ls='--', label='gpt2', element='step', stat='density', ax=ax[2, 1])
    ax[2,1].set_ylabel('')
    ax[2,1].set_xlabel(r'$\sigma_\varphi$')

    custom_ticks = {
        0: [0, 0.2, 0.4, 0.6, 0.8, 1],
        1: [-8, -4, 0, 4, 8],
        2: [0, 0.1, 0.2, 0.3, 0.4, 0.5] ,
        3: [-8, -4, 0, 4, 8],
        4: [0, 0.1, 0.2, 0.3, 0.4],
        5: [0.5, 0.75, 1, 1.25, 1.5]
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
    ax[0,1].set_xlim(-8, 8)
    ax[1,1].set_xlim(-8, 8)
    ax[2,0].set_xlim(0, 0.4)
    ax[2,1].set_xlim(0.5, 1.5)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.65, wspace=0.3) 
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()



def ordered_z_plots(toy_qcd, gen_jets):

    fig, ax = plt.subplots(3, 2, figsize=(6, 6))

    bins=np.arange(0, 1, 0.01)

    # --- Plot hardest ---

    sns.histplot(toy_qcd[:,0,:][...,0], bins=bins, alpha=0.25, color='crimson', lw=0, label='data', element='step', stat='density', ax=ax[0, 0])
    sns.histplot(gen_jets[:,0,:][...,0], bins=bins,  fill=False, color='crimson', lw=1, label='gpt2', element='step', stat='density', ax=ax[0, 0])

    ax[0,0].set_ylabel('Density')
    ax[0,0].set_xlabel(r'hardest $z$')
    ax[0,0].legend(loc='upper left', fontsize=6)
    ax[0,0].set_xlim(0, 1)


    sns.histplot(toy_qcd[:,5,:][...,0], bins=bins,   alpha=0.25, color='crimson', lw=0, label='B', element='step', stat='density', ax=ax[0, 1])
    sns.histplot(gen_jets[:,5,:][...,0], bins=bins,  fill=False, color='crimson', lw=1, label='gpt2', element='step', stat='density', ax=ax[0, 1])
    ax[0,1].set_ylabel('')
    ax[0,1].set_xlabel(r'5th hardest $z$')


    bins=np.arange(0, 0.6, 0.01)

    sns.histplot(toy_qcd[:,10,:][...,0], bins=bins,  alpha=0.25, color='crimson', lw=0, label='B', element='step', stat='density', ax=ax[1, 0])
    sns.histplot(gen_jets[:,10,:][...,0], bins=bins,  fill=False, color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[1, 0])

    ax[1,0].set_ylabel('Density')
    ax[1,0].set_xlabel(r'10th hardest $z$')
    # ax[1,0].legend(loc='upper right', fontsize=6)
    ax[1,0].set_xlim(0, 0.6)

    bins=np.arange(0, 0.6, 0.008)

    sns.histplot(toy_qcd[:,15,:][...,0], bins=bins,  alpha=0.25, color='crimson', lw=0, label='B', element='step', stat='density', ax=ax[1, 1])
    sns.histplot(gen_jets[:,15,:][...,0], bins=bins, fill=False, color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[1, 1])
    ax[1,1].set_ylabel('')
    ax[1,1].set_xlabel(r'15th hardest $z$')
    # ax[1,1].legend(loc='upper right', fontsize=6)
    ax[1,1].set_xlim(0, 0.6)


    # Middle row: intermediate distributions

    bins=np.arange(0, 0.2, 0.004)

    sns.histplot(toy_qcd[:,20,:][...,0], bins=bins,  alpha=0.25, color='crimson', lw=0, label='B', element='step', stat='density', ax=ax[2, 0])
    sns.histplot(gen_jets[:,20,:][...,0], bins=bins, fill=False,  color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[2, 0])

    ax[2,0].set_ylabel('Density')
    ax[2,0].set_xlabel(r' 20th hardest $z$')
    # ax[2,0].legend(loc='upper right', fontsize=6)
    ax[2,0].set_xlim(0, 0.4)


    bins=np.arange(0, 0.05, 0.001)

    sns.histplot(toy_qcd[:,25,:][...,0], bins=bins,  alpha=0.25, color='crimson', lw=0, label='B', element='step', stat='density', ax=ax[2, 1])
    sns.histplot(gen_jets[:,25,:][...,0], bins=bins, fill=False,  color='crimson', lw=1, label='S', element='step', stat='density', ax=ax[2, 1])
    ax[2,1].set_ylabel('')
    ax[2,1].set_xlabel(r'25th hardest $z$')
    # ax[2,1].legend(loc='upper right', fontsize=6)
    ax[2,1].set_xlim(0, 0.1)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.65, wspace=0.3) 
    plt.show()