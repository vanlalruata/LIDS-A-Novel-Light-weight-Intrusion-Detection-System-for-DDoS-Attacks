from os import listdir
from os import walk
import warnings
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def get_file_names(PATH):
    """
    The function returns the filenames of all the datasets in the dataset folder
    Args:
        PATH: path of the dataset folder
    Return:
        filenames: List of the files in the dataset folder
    """
    print("**********************************Loading Filenames****************************************")
    filenames = next(walk(PATH), (None, None, []))[2]
    print("Files Found in Dataset Folder")
    for i in range(len(filenames)):
        print("{}. {}".format(i + 1, filenames[i]))

    return filenames


def load_files(PATH, filenames, nrows):
    """
    1. Loading the datafiles individually
    2. Mapping attack and non attack packets
    3. concatenating them
    Args:
        PATH: Dataset Path
        filenames: names of the files in the dataset
        meaning_less_cols: columns that are not needed
    Return:
        dataset
    """

    print("************************************Loading Files******************************************")
    i = 0
    for item in filenames:

        item = PATH + '/' + item

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if i == 0:
                dataset = pd.read_csv(item, nrows=nrows)
            else:
                df1 = pd.read_csv(item, nrows=nrows)
                dataset = pd.concat([dataset, df1])
                del df1
            i = i + 1

    dataset[' Label'] = dataset[' Label'].map(lambda x: 0 if x == 'BENIGN' else 1)

    print('{} Files Loaded Sucessfully'.format(i))
    print("Dataset Shape", dataset.shape)
    print("***************************Loading Files Completed******************************************")

    return dataset


def drop_meaningless_cols(dataset, meaning_less_cols):
    """
    The function drops the meaning less columns in the dataset
    Args:
        datset: pandas dataset
        meaning_less_cols: cols not needed
    Return:
        dataset: removed useless features
    """
    print("*****************************Delete Meaningless Features**********************************")
    dataset.drop(meaning_less_cols, axis=1, inplace=True)

    for i in range(len(meaning_less_cols)):
        print("{}. {}".format(i + 1, meaning_less_cols[i]))

    print("Dataset Shape: ", dataset.shape)
    print("*****************************Deleted Meaningless Features**********************************")
    return dataset


def drop_constant_features(dataset):
    """
    Drop columns having only one type of value
    Args:
        dataset: pandas dataframe
    Return:
        dataset: pandas dataframe
    """
    print("*******************************Drop Constant Features**************************************")
    columns_with_one_value = []
    for col in dataset.columns:
        if len(dataset[col].unique()) == 1:
            columns_with_one_value.append(col)

    dataset.drop(columns_with_one_value, axis=1, inplace=True)

    print("Dropped Constant Features:")
    for i in range(len(columns_with_one_value)):
        print("{}. {}".format(i + 1, columns_with_one_value[i]))

    print("Dataset Shape:", dataset.shape)
    print("*******************************Droped Constant Features**************************************")

    return dataset


def min_max_scaler(dataset):
    scaler = MinMaxScaler()

    dataset = dataset.reset_index()
    dataset.drop('index', axis=1, inplace=True)

    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis=1, inplace=True)
    features = dataset1.columns

    dataset1 = scaler.fit_transform(dataset1)

    dataset1 = pd.DataFrame(dataset1, columns=features)

    dataset1[' Label'] = dataset[' Label']

    return dataset1


def drop_duplicate_features(dataset):
    """
    Finds the duplicate features in dataset and drops them
    Args:
        dataset: pandas dataframe
    Return:
        dataset: modified dataset
    """

    print("******************************Drop Duplicate Features***************************************")
    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis=1, inplace=True)

    dataset1 = dataset1.T.drop_duplicates().T

    dataset1[' Label'] = dataset[' Label']

    print("Droped Duplicated Features:")

    droped_features = set(dataset.columns) - set(dataset1.columns)
    i = 1
    for item in droped_features:
        print("{}. {}".format(i, item))
        i += 1

    print("Dataset Shape:", dataset1.shape)

    print("******************************Droped Duplicate Features***************************************")
    return dataset1
