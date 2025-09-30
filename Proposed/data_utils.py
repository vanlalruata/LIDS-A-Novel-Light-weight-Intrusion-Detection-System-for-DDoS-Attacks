from LIDS.data_utils import get_file_names, drop_meaningless_cols, drop_constant_features, min_max_scaler
from sklearn.preprocessing import LabelEncoder
from info_gain import info_gain
from LIDS.data_utils import min_max_scaler
from sklearn.feature_selection import VarianceThreshold
import warnings
import numpy as np
import pandas as pd
import os
import pdb
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch


def weighted_random_sampler(train):
    '''
    Args:
      train: train dataset (must have attribute `labels` as a 1D torch tensor of class ids)
    return:
      weighted_sampler: sampler to draw samples inversely proportional to class frequency.
    '''

    # Ensure we operate on CPU numpy ints
    targets_t = train.labels
    if isinstance(targets_t, torch.Tensor):
        targets_np = targets_t.detach().cpu().numpy().astype(int)
    else:
        targets_np = np.asarray(train.labels).astype(int)

    # Compute class weights using actual class ids (not assuming 0..K-1)
    classes, counts = np.unique(targets_np, return_counts=True)

    # Edge case: if only one class present, use uniform weights to avoid index errors
    if len(classes) == 1:
        samples_weight = np.ones_like(targets_np, dtype=np.float64)
    else:
        class_weight = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        samples_weight = np.array([class_weight[c] for c in targets_np], dtype=np.float64)

    samples_weight_t = torch.from_numpy(samples_weight)
    weighted_sampler = WeightedRandomSampler(samples_weight_t, len(samples_weight_t))

    return weighted_sampler


def make_datasets(PATH, nrows):
    """
    The function loads the processed dataset file and does
    1. inf records are deleted
    2. nan values deleted
    3. constant features removed
    4. duplicate features removed
    5. duplicates removed
    6. scaling
    7. less info gain features removed
    8. qasi constant features removed
    
    """

    dataset = load_dataset(PATH, nrows)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    dataset.dropna(axis=0, inplace=True)
    print('Drop na ', dataset.shape)
    dataset = drop_constant_features(dataset)
    dataset = drop_duplicate_features(dataset)
    print("Dataset Shape:", dataset.shape)
    dataset.drop_duplicates(inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    print("drop duplicates Dataset Shape:", dataset.shape)

    dataset = min_max_scaler(dataset)
    dataset = drop_less_information_gain_features(dataset)
    dataset = drop_qasi_constant_features(dataset)

    dataset.to_csv(os.getcwd() + '/Datasets/Processed_Dataset.csv')

    # print(mappings)
    print("Dataset Shape", dataset.shape)
    print("********************************* Dataset Created*****************************************")


def load_dataset(PATH, nrows):
    """
    1. load an individual file
    2. drop na and duplicates
    3. Concnate with the previous file that is loaded
    """

    # filenames= get_file_names(PATH)
    meaning_less_cols = ['Unnamed: 0', 'Flow ID', ' Timestamp', ' Source IP',
                         'SimillarHTTP', ' Source Port', ' Destination IP', ' Destination Port']

    filenames = ['TFTP.csv', 'DrDoS_SNMP.csv', 'DrDoS_DNS.csv', 'DrDoS_MSSQL.csv', 'DrDoS_SSDP.csv',
                 'DrDoS_NetBIOS.csv', 'DrDoS_LDAP.csv', 'DrDoS_NTP.csv', 'Syn.csv', 'UDPLag.csv', 'DrDoS_UDP.csv']

    print("************************************Loading Files******************************************")
    i = 0
    for item in filenames:
        print('*****************************************************************************************')
        print('*****************************************************************************************')

        print('File to Load: ', item)

        item = PATH + '/' + item
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if i == 0:

                dataset = pd.read_csv(item, nrows=nrows)
                print("*****************Original File Shape******************")
                print(dataset.shape)

                dataset = drop_meaningless_cols(dataset, meaning_less_cols)

                dataset.drop_duplicates(inplace=True)
                dataset.dropna(axis=0, inplace=True)
                print("After null and duplicate: ", dataset.shape)

            else:

                df1 = pd.read_csv(item, nrows=nrows)
                print("*****************Original File Shape***************")
                print(df1.shape)

                df1 = drop_meaningless_cols(df1, meaning_less_cols)
                df1.drop_duplicates(inplace=True)
                df1.dropna(axis=0, inplace=True)

                print("After null and duplicate:", df1.shape)

                dataset = pd.concat([dataset, df1])
                print("############After Concnation Dataset Shape############## ")
                print(dataset.shape)
                dataset.drop_duplicates(inplace=True)

                print('After duplicate Removal:', dataset.shape)

                del df1

            i = i + 1

    print("***************After loading all files Dataset Shape***********:")
    print(dataset.shape)

    print("************************************ Files Loaded ******************************************")

    # dataset.to_csv('full_dataset.csv')

    return dataset


def drop_duplicate_features(dataset):
    print("*********************************Dropping Duplicate Features *****************************************")

    dup_features = []
    features = dataset.columns
    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            x = dataset[features[i]] == dataset[features[j]]
            if x.sum() == len(x):
                dup_features.append(features[j])

    for i in range(len(dup_features)):
        print("{}. {}".format(i + 1, dup_features[i]))

    dataset.drop(dup_features, axis=1, inplace=True)

    print("*********************************Dropped Duplicate Features *****************************************")
    print("Dataset Shape: ", dataset.shape)
    return dataset


def drop_less_information_gain_features(dataset, threshold=0.05):
    '''
    Measures the reduction in entropy after the split  
    
    '''
    print("*******************************Deleting less info_gain features*************************")

    less_ig_cols = []

    features = list(set(dataset.columns) - set([' Label']))
    for col in features:
        info_gain_value = info_gain.info_gain(dataset[col], dataset[' Label'])
        if info_gain_value < threshold:
            less_ig_cols.append(col)
    i = 1
    for item in less_ig_cols:
        print("{}. {}".format(i, item))
        i = i + 1

    print("*******************************Less info_gain features Deleted*************************")

    dataset.drop(less_ig_cols, axis=1, inplace=True)
    print("Dataset Shape", dataset.shape)

    return dataset


def load_processed_dataset():
    print("********************************* Loading Preprocessed dataset********************************")

    PATH = os.getcwd() + '/Datasets/Processed_Dataset.csv'
    dataset = pd.read_csv(PATH)
    dataset.drop('Unnamed: 0', axis=1, inplace=True)

    print("************************************ Preprocessed dataset Loaded****************************")

    return dataset


def load_pca_dataset():
    print("********************************* Loading PCA dataset********************************")

    base_path = os.getcwd() + '/Datasets/'
    pca_path = os.path.join(base_path, 'PCA_Dataset.csv')
    train_path = os.path.join(base_path, 'train_PCA_Dataset.csv')
    test_path = os.path.join(base_path, 'test_PCA_Dataset.csv')

    dataset = None
    if os.path.exists(pca_path):
        dataset = pd.read_csv(pca_path)
    else:
        parts = []
        if os.path.exists(train_path):
            parts.append(pd.read_csv(train_path))
        if os.path.exists(test_path):
            parts.append(pd.read_csv(test_path))
        if parts:
            dataset = pd.concat(parts, ignore_index=True)

    if dataset is None:
        raise FileNotFoundError("No PCA dataset found. Please run option 2 to create PCA datasets first.")

    # Drop index column if present
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop('Unnamed: 0', axis=1, inplace=True)

    print("************************************ PCA dataset Loaded****************************")

    return dataset


def create_pca_dataset(components=41):
    print("****************************** Creating PCA Datasets *******************************")
    from sklearn.decomposition import PCA
    dataset = load_processed_dataset()
    encoder = LabelEncoder()
    dataset[' Label'] = encoder.fit_transform(dataset[' Label'])
    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis=1, inplace=True)

    # Ensure the number of PCA components does not exceed the available features
    n_features = dataset1.shape[1]
    if components is None or components <= 0:
        components = n_features
    if components > n_features:
        print(f"Requested PCA components ({components}) exceed available features ({n_features}). Reducing to {n_features}.")
        components = n_features

    pca = PCA(n_components=components)

    column_names = [f'PC {i + 1}' for i in range(components)]

    principal_components = pca.fit_transform(dataset1)

    principal_components = pd.DataFrame(principal_components, columns=column_names)

    train_PCA, test_PCA, train_label, test_label = train_test_split(principal_components, dataset[' Label'],
                                                                    test_size=0.33, random_state=42)

    train_PCA[' Label'] = train_label
    test_PCA[' Label'] = test_label

    print("****************************** PCA Datasets Created *******************************")
    # principal_components.to_csv(os.getcwd()+'/Datasets/PCA_Dataset.csv')
    train_PCA.to_csv(os.getcwd() + '/Datasets/train_PCA_Dataset.csv')
    test_PCA.to_csv(os.getcwd() + '/Datasets/test_PCA_Dataset.csv')
    print("****************************** Created PCA Datasets Saved *******************************")


def drop_qasi_constant_features(dataset):
    print("************************************ Dropping Qasi Constant Features ****************************")

    dataset1 = dataset.copy()
    VarianceThreshold
    dataset1.drop(' Label', axis=1, inplace=True)
    import pdb

    qasi_constant_filter = VarianceThreshold(threshold=0.01)

    qasi_constant_filter.fit(dataset1)

    qasi_support = qasi_constant_filter.get_support()

    qasi_constant_features = []
    features = dataset1.columns
    for i in range(len(qasi_support)):
        if qasi_support[i] == True:
            qasi_constant_features.append(features[i])

    for i in range(len(qasi_constant_features)):
        print("{}. {}".format(i + 1, qasi_constant_features[i]))

    dataset.drop(qasi_constant_features, axis=1, inplace=True)

    print("************************************ Dropped Qasi Constant Features ****************************")

    return dataset
