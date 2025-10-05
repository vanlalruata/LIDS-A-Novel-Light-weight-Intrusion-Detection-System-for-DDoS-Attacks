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

    # Get prefix
    prefix = 'CICDDoS2019'
    try:
        with open(os.path.join(os.getcwd(), 'Datasets', 'current_dataset_prefix.txt'), 'r', encoding='utf-8') as f:
            s = f.read().strip()
            if s:
                prefix = s
    except Exception:
        pass

    # Combined plot: Explained variance and Cumulative variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Explained variance (bar chart)
    ax1.set_title('Explained Variance Across Principal Components', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Principal Components', fontsize=12)
    ax1.set_ylabel('Explained Variance', fontsize=12)
    sns.barplot(x=idx[:limit_df], y='explained variance', data=explained_variance_limited, color='tab:green', ax=ax1)
    ax1.axhline(mean_explained_variance, ls='--', color='grey')
    ax1.text(1.5, mean_explained_variance + (mean_explained_variance * .05), "Average", color='black', fontsize=12)
    max_y1 = max(explained_variance_limited.iloc[:, 0])
    ax1.set(ylim=(0, max_y1 + max_y1 * .1))
    ax1.grid(True, alpha=0.3)
    
    # Right: Cumulative explained variance (line chart)
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Principal Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    sns.lineplot(x=idx[:limit_df], y='cumulative', data=explained_variance_limited, color='dodgerblue', linewidth=2, ax=ax2)
    max_y2 = max(explained_variance_limited.iloc[:, 1])
    ax2.set(ylim=(0, max_y2 + max_y2 * .1))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'Images', f'{prefix}_pca_variance_analysis.png'), dpi=600)
    print(f"Saved PCA variance analysis to Images/{prefix}_pca_variance_analysis.png")
    plt.close(fig)

    print("******************************PCA Done*******************************")


def plot_pca_analysis(dataset):
    import re, seaborn as sns, numpy as np, pandas as pd, random

    from matplotlib.pyplot import plot, show, draw, figure, cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import LabelEncoder

    sns.set_style("whitegrid", {'axes.grid': False})
    
    prefix = 'CICDDoS2019'
    try:
        with open(os.path.join(os.getcwd(), 'Datasets', 'current_dataset_prefix.txt'), 'r', encoding='utf-8') as f:
            s = f.read().strip()
            if s:
                prefix = s
    except Exception:
        pass

    # Multiclass PCA Analysis
    fig_multi = plt.figure(figsize=(10, 8))
    ax_multi = fig_multi.add_subplot(111, projection='3d')

    colors = ['orange', 'dodgerblue', 'limegreen', 'red', 'purple',
              'gold', 'hotpink', 'cyan', 'coral', 'teal',
              'magenta', 'yellowgreen', 'navy']
    classses = dataset[' Label'].unique()
    markers = ['+', '*', 'v', '<', '>', 'X', 'D', 'h', 'x', 'p', 'P', '8', 's']

    for i in range(len(classses)):
        df = dataset[dataset[' Label'] == classses[i]]
        pc1 = df['PC 1']
        pc2 = df['PC 2']
        pc3 = df['PC 3']
        marker_idx = i % len(markers)
        color_idx = i % len(colors)
        ax_multi.scatter(pc2, pc1, pc3, marker=markers[marker_idx], s=6, color=colors[color_idx])

    ax_multi.set_xlabel('PC 2')
    ax_multi.set_ylabel('PC 1')
    ax_multi.set_zlabel('PC 3')
    ax_multi.set_title('Multiclass PCA Analysis')
    ax_multi.legend(classses, loc=1, fontsize=8)
    ax_multi.set_xlim3d(min(dataset['PC 2']), max(dataset['PC 2']))
    ax_multi.set_ylim3d(min(dataset['PC 1']), max(dataset['PC 1']))
    ax_multi.set_zlim3d(min(dataset['PC 3']), max(dataset['PC 3']))
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'Images', f'{prefix}_pca_multiclass_analysis.png'), dpi=600)
    print(f"Saved multiclass PCA analysis to Images/{prefix}_pca_multiclass_analysis.png")
    plt.close(fig_multi)

    # Binary PCA Analysis
    dataset_binary = dataset.copy()
    dataset_binary[' Label'] = dataset_binary[' Label'].map(lambda a: 'Normal' if str(a).upper() in ['BENIGN', 'NORMAL'] else 'Attack')

    fig_binary = plt.figure(figsize=(10, 8))
    ax_binary = fig_binary.add_subplot(111, projection='3d')
    
    colors_bin = ['orange', 'red']
    classses_bin = dataset_binary[' Label'].unique()
    markers_bin = ['.', '*']

    for i in range(len(classses_bin)):
        df = dataset_binary[dataset_binary[' Label'] == classses_bin[i]]
        pc1 = df['PC 1']
        pc2 = df['PC 2']
        pc3 = df['PC 3']
        ax_binary.scatter(pc1, pc2, pc3, marker=markers_bin[i], s=6, color=colors_bin[i])

    ax_binary.set_xlabel('PC 1')
    ax_binary.set_ylabel('PC 2')
    ax_binary.set_zlabel('PC 3')
    ax_binary.set_title('Binary PCA Analysis')
    ax_binary.legend(classses_bin, loc=1)
    ax_binary.set_xlim3d(min(dataset_binary['PC 1']), max(dataset_binary['PC 1']))
    ax_binary.set_ylim3d(min(dataset_binary['PC 2']), max(dataset_binary['PC 2']))
    ax_binary.set_zlim3d(min(dataset_binary['PC 3']), max(dataset_binary['PC 3']))

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'Images', f'{prefix}_pca_binary_analysis.png'), dpi=600)
    print(f"Saved binary PCA analysis to Images/{prefix}_pca_binary_analysis.png")
    plt.close(fig_binary)
