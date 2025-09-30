from matplotlib import pyplot as plt
import pandas as pd
import os
from pylab import *

plt.rcParams["font.serif"] = "Times New Roman"


def class_distributions(dataset):
    items = dataset[' Label'].value_counts()

    keys = []
    values = []
    for key, value in items.items():
        keys.append(key)
        values.append(value)

    expl = [0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.65]
    pctdist = 0.85

    ax = plt.subplot2grid((1, 2), (0, 0))

    patches, texts, autotexts = plt.pie(values, explode=expl, autopct='%1.2f%%', pctdistance=pctdist,
                                        wedgeprops={"edgecolor": "black",
                                                    'linewidth': 0.2,
                                                    'antialiased': True}, textprops={'fontsize': 6}, startangle=40)

    i = 0
    for patch, txt in zip(patches, autotexts):
        ang = (patch.theta2 + patch.theta1) / 2.

        x = (1.2 + expl[i]) * np.cos(ang * np.pi / 180)
        y = (1.2 + expl[i]) * np.sin(ang * np.pi / 180)

        if i == 11:
            y = y - 0.05
        if i > 10:
            x = x - 0.05

        if (patch.theta2 - patch.theta1) < 5:
            txt.set_position((x, y - 0.1))
        i = i + 1
    plt.title(' Multi Class Distributions', size=8)

    plt.legend(keys, fontsize=4, bbox_to_anchor=(-0.11, 0.7, 0.3, 0.5))

    centre_circle = plt.Circle((0, 0), 0.30, fc='white')

    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    autoAxis = ax.axis()
    rec = Rectangle((autoAxis[0] - 0.3, autoAxis[2]), (autoAxis[1] - autoAxis[0]) + 0.55,
                    (autoAxis[3] - autoAxis[2]) + 0.5, fill=False, lw=1, linestyle='dashed')
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)

    labels = ['Normal', 'Attack']
    normal = dataset[dataset[' Label'] == 'BENIGN'][' Label'].count()
    attack = dataset[dataset[' Label'] != 'BENIGN'][' Label'].count()

    ax = plt.subplot2grid((1, 2), (0, 1))

    plt.pie([normal, attack], explode=[0.051, 0.05], autopct='%1.2f%%', pctdistance=0.85,
            wedgeprops={"edgecolor": "black",
                        'linewidth': 0.2,
                        'antialiased': True}, textprops={'fontsize': 6})

    plt.title('Binary Class Distributions', size=8)
    plt.legend(labels, fontsize=4)

    centre_circle = plt.Circle((0, 0), 0.30, fc='white')

    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    autoAxis = ax.axis()
    rec = Rectangle((autoAxis[0] - 0.3, autoAxis[2]), (autoAxis[1] - autoAxis[0]) + 0.55,
                    (autoAxis[3] - autoAxis[2]) + 0.5, fill=False, lw=1, linestyle='dashed')
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)

    plt.subplots_adjust(left=0.06,
                        bottom=0,
                        right=0.95,
                        top=0.99,
                        wspace=0.2149,
                        )

    plt.savefig(os.getcwd() + '/Images/class_distribution.png', dpi=600)


def pca_analysis(dataset, components=3):
    print("****************************** PCA *******************************")
    from sklearn.decomposition import PCA
    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis=1, inplace=True)
    pca = PCA(n_components=components)

    principal_components = pca.fit_transform(dataset1)

    principal_components = pd.DataFrame(principal_components, columns=['PC 1', 'PC 2', 'PC 3'])

    principal_components[' Label'] = dataset[' Label']
    print("****************************** PCA Done *******************************")

    return principal_components


def plot_legend_files(i, j):
    """
    Plots legend for the plot_files_processing int the last subplot
    """
    ax = plt.subplot2grid((3, 4), (i, j))
    ax.add_patch(plt.Rectangle((0.2, 0.1),
                               0.7, 0.4,
                               fc='orange',
                               ec='black',
                               lw=1))
    ax.add_patch(plt.Rectangle((0.2, 0.5),
                               0.7, 0.4,
                               fc='mediumspringgreen',
                               ec='black',
                               lw=1))
    plt.text(0.2, 0.3, ' After Processing', fontsize=7)
    plt.text(0.3, 0.7, 'Deleted', fontsize=8)
    plt.axis('off')


def plot_files_preprocessing():
    """
    Plots pie charts of the individual files describing the amount of data deleted after
    nan deletion, drop duplicates and meaningless col deletion
    """
    filenames = ['TFTP', 'DrDoS SNMP', 'DrDoS DNS', 'DrDoS MSSQL', 'DrDoS SSDP',
                 'DrDoS NetBIOS', 'DrDoS LDAP', 'DrDoS NTP', 'Syn', 'UDPLag', 'DrDoS UDP', ' ']

    # The values are got after the final file was created and the sizes were recorded
    original_size = [20107827, 5161377, 5074413, 4524498, 2611374, 4094986,
                     2181542, 1217007, 1582681, 370605, 3136802, 0]
    after_size = [4419736, 115489, 116313, 208880, 891959, 21124,
                  31501, 1126258, 155893, 93021, 1077387, 0]
    deleted = []
    for i in range(len(original_size)):
        deleted.append(original_size[i] - after_size[i])

    labels = ['After Processing', 'Deleted']
    expl = [0.05, 0.05]
    index = 0
    pctdist = 0.6

    for i in range(0, 3):
        for j in range(0, 4):
            if index == 11:
                plot_legend_files(i, j)
                break
            ax = plt.subplot2grid((3, 4), (i, j))
            plt.rcParams["font.serif"] = "Times New Roman"

            patches, texts, autotexts = plt.pie([after_size[index], deleted[index]], explode=expl, autopct='%1.2f%%',
                                                pctdistance=pctdist,
                                                wedgeprops={"edgecolor": "black",
                                                            'linewidth': 0.2,
                                                            'antialiased': True}, textprops={'fontsize': 6},
                                                colors=['orange', 'mediumspringgreen'])

            #####################################

            for patch, txt in zip(patches, autotexts):
                ang = (patch.theta2 + patch.theta1) / 2.

                x = (1 * np.cos(ang * np.pi / 180))
                y = (1 * np.sin(ang * np.pi / 180))

                if (patch.theta2 - patch.theta1) < 20:
                    txt.set_position((x - 0.4, y + 0.1))

            plt.title(filenames[index], size=6)

            centre_circle = plt.Circle((0, 0), 0.30, fc='white')

            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)

            autoAxis = ax.axis()
            rec = Rectangle((autoAxis[0] - 0.3, autoAxis[2]), (autoAxis[1] - autoAxis[0]) + 0.37,
                            (autoAxis[3] - autoAxis[2]) + 0.512, fill=False, lw=1, linestyle='dashed')
            rec = ax.add_patch(rec)
            rec.set_clip_on(False)

            index += 1
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(os.getcwd() + '/Images/files_distribution.png', dpi=600)


def plot_proposed_model_accuracy_loss():
    data = pd.read_csv(os.getcwd() + '/Datasets/train_val_metrics.csv')

    # Accuracy figure
    plt.figure(figsize=(8, 5))
    plt.plot(data['Unnamed: 0'], data['Train Accuracy'], color='grey', label='Train')
    plt.plot(data['Unnamed: 0'], data['Validation Accuracy'], color='black', label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.getcwd() + '/Images/proposed_model_accuracy.png', dpi=600)
    plt.close()

    # Loss figure
    plt.figure(figsize=(8, 5))
    plt.plot(data['Unnamed: 0'], data['Train Loss'], color='grey', label='Train')
    plt.plot(data['Unnamed: 0'], data['Validation Loss'], color='black', label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.getcwd() + '/Images/proposed_model_loss.png', dpi=600)
    plt.close()


def plot_multi_proposed_model_accuracy_loss():
    data = pd.read_csv(os.getcwd() + '/Datasets/train_val_metrics_multi.csv')

    # Accuracy figure
    plt.figure(figsize=(8, 5))
    plt.plot(data['Unnamed: 0'], data['Train Accuracy'], color='grey', label='Train')
    plt.plot(data['Unnamed: 0'], data['Validation Accuracy'], color='black', label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.getcwd() + '/Images/proposed_model_multi_accuracy.png', dpi=600)
    plt.close()

    # Loss figure
    plt.figure(figsize=(8, 5))
    plt.plot(data['Unnamed: 0'], data['Train Loss'], color='grey', label='Train')
    plt.plot(data['Unnamed: 0'], data['Validation Loss'], color='black', label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.getcwd() + '/Images/proposed_model_multi_loss.png', dpi=600)
    plt.close()


plot_multi_proposed_model_accuracy_loss()