from LIDS.Proposed.data_utils import load_processed_dataset
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os


def principal_component_varience():
    print("****************************** PCA Started*******************************")

    dataset = load_processed_dataset()

    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis=1, inplace=True)
    pca = PCA()

    principal_components = pca.fit_transform(dataset1)

    n_components = len(pca.explained_variance_ratio_)
    explained_variance = pca.explained_variance_ratio_
    cum_explained_variance = np.cumsum(explained_variance)
    idx = np.arange(n_components) + 1
    explained_variance = pd.DataFrame([explained_variance, cum_explained_variance],
                                      index=['explained variance', 'cumulative'],
                                      columns=idx).T
    mean_explained_variance = explained_variance.iloc[:, 0].mean()

    limit_df = 10
    explained_variance_limited = explained_variance.iloc[:limit_df, :]

    # Plot 1: Explained variance (bar chart)
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.set_title('Explained Variance Across Principal Components', fontsize=14)
    ax1.set_xlabel('Principal Components', fontsize=12)
    ax1.set_ylabel('Explained Variance', fontsize=12)
    sns.barplot(x=idx[:limit_df], y='explained variance', data=explained_variance_limited, color='tab:green', ax=ax1)
    ax1.axhline(mean_explained_variance, ls='--', color='grey')  # plot mean
    ax1.text(1.5, mean_explained_variance + (mean_explained_variance * .05), "Average", color='black', fontsize=14)
    max_y1 = max(explained_variance_limited.iloc[:, 0])
    ax1.set(ylim=(0, max_y1 + max_y1 * .1))
    plt.tight_layout()
    prefix = 'CICDDoS2019'
    try:
        with open(os.path.join(os.getcwd(), 'Datasets', 'current_dataset_prefix.txt'), 'r', encoding='utf-8') as f:
            s = f.read().strip()
            if s:
                prefix = s
    except Exception:
        pass
    plt.savefig(os.path.join(os.getcwd(), 'Images', f'{prefix}_explained_varience_pca.png'), dpi=600)
    plt.close(fig1)

    # Plot 2: Cumulative explained variance (line chart)
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.set_title('Cumulative Explained Variance', fontsize=14)
    ax2.set_xlabel('Principal Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    sns.lineplot(x=idx[:limit_df], y='cumulative', data=explained_variance_limited, color='black', ax=ax2)
    max_y2 = max(explained_variance_limited.iloc[:, 1])
    ax2.set(ylim=(0, max_y2 + max_y2 * .1))
    plt.tight_layout()
    prefix = 'CICDDoS2019'
    try:
        with open(os.path.join(os.getcwd(), 'Datasets', 'current_dataset_prefix.txt'), 'r', encoding='utf-8') as f:
            s = f.read().strip()
            if s:
                prefix = s
    except Exception:
        pass
    plt.savefig(os.path.join(os.getcwd(), 'Images', f'{prefix}_cumulative_explained_varience_pca.png'), dpi=600)
    plt.close(fig2)

    print("******************************PCA Done*******************************")


def plot_pca_analysis(dataset):
    import re, seaborn as sns, numpy as np, pandas as pd, random

    from matplotlib.pyplot import plot, show, draw, figure, cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import LabelEncoder

    sns.set_style("whitegrid", {'axes.grid': False})

    fig, ax = plt.subplots(1, 2, figsize=(15, 10), subplot_kw=dict(projection='3d'))

    colors = ['black', 'grey', 'orange', 'yellowgreen', 'steelblue',
              'violet', 'hotpink', 'purple', 'lightcoral', 'cadetblue',
              'blueviolet', 'orchid', 'cyan']
    classses = dataset[' Label'].unique()
    markers = ['+', '*', 'v', '<', '>', 'X', 'D', 'h', 'x', 'p', 'P', '8', 's']

    for i in range(len(classses)):
        df = dataset[dataset[' Label'] == classses[i]]
        pc1 = df['PC 1']
        pc2 = df['PC 2']
        pc3 = df['PC 3']
        print(i)
        ax[0].scatter(pc2, pc1, pc3, marker=markers[i], s=6, color=colors[i])

    ax[0].set_xlabel('PC 2')
    ax[0].set_ylabel('PC 1')
    ax[0].set_zlabel('PC 3')
    ax[0].set_title('Multi Class Analysis')
    ax[0].legend(classses, loc=1)
    ax[0].set_xlim3d(min(dataset['PC 2']), max(dataset['PC 2']))
    ax[0].set_ylim3d(min(dataset['PC 1']), max(dataset['PC 1']))
    ax[0].set_zlim3d(min(dataset['PC 3']), max(dataset['PC 3']))

    dataset[' Label'] = dataset[' Label'].map(lambda a: 'Normal' if a == 'BENIGN' else 'Attack')

    colors = ['orange', 'red']
    classses = dataset[' Label'].unique()
    markers = ['.', '*']

    for i in range(len(classses)):
        df = dataset[dataset[' Label'] == classses[i]]
        pc1 = df['PC 1']
        pc2 = df['PC 2']
        pc3 = df['PC 3']
        ax[1].scatter(pc1, pc2, pc3, marker=markers[i], s=6, color=colors[i])

    ax[1].set_xlabel('PC 1')
    ax[1].set_ylabel('PC 2')
    ax[1].set_zlabel('PC 3')
    ax[1].set_title('Multi Class Analysis')
    ax[1].legend(classses, loc=1)
    ax[1].set_xlim3d(min(dataset['PC 1']), max(dataset['PC 1']))
    ax[1].set_ylim3d(min(dataset['PC 2']), max(dataset['PC 2']))
    ax[1].set_zlim3d(min(dataset['PC 3']), max(dataset['PC 3']))

    plt.subplots_adjust(left=0.05,
                        bottom=0,
                        right=0.95,
                        top=1,
                        wspace=0.1,
                        )

    prefix = 'CICDDoS2019'
    try:
        with open(os.path.join(os.getcwd(), 'Datasets', 'current_dataset_prefix.txt'), 'r', encoding='utf-8') as f:
            s = f.read().strip()
            if s:
                prefix = s
    except Exception:
        pass
    plt.savefig(os.path.join(os.getcwd(), 'Images', f'{prefix}_pca_analysis.png'), dpi=600)
