from torch.utils.data import Dataset
from LIDS.Proposed.data_utils import weighted_random_sampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import pandas as pd
import os
import numpy as np





class CIC2019(Dataset):
    
    def __init__(self, kind='train', n_features=10):
        if kind=='train':
            print("******************** Loading Train PCA Dataset **********************")
            xy = pd.read_csv(os.getcwd()+'/Datasets/train_PCA_Dataset.csv')
            # Ensure binary labels 0/1, even if numeric multiclass ids are present
            if xy[' Label'].dtype == object:
                xy[' Label'] = xy[' Label'].apply(map_binary_class_attack)
            else:
                uniq = set(pd.unique(xy[' Label']))
                if not uniq.issubset({0, 1}):
                    xy[' Label'] = xy[' Label'].apply(lambda v: 0 if int(v) == 0 else 1)
            
            self.labels = torch.Tensor(xy[' Label'])
            # Drop auto index column if present
            drop_cols = [c for c in ['Unnamed: 0', ' Label'] if c in xy.columns]
            xy.drop(drop_cols, axis=1, inplace=True)
            print("******************** Train PCA Dataset Loaded **********************")
    
        else:
            print("******************** Loading Test PCA Dataset **********************")

            xy = pd.read_csv(os.getcwd()+'/Datasets/test_PCA_Dataset.csv')
            # Ensure binary labels 0/1, even if numeric multiclass ids are present
            if xy[' Label'].dtype == object:
                xy[' Label'] = xy[' Label'].apply(map_binary_class_attack)
            else:
                uniq = set(pd.unique(xy[' Label']))
                if not uniq.issubset({0, 1}):
                    xy[' Label'] = xy[' Label'].apply(lambda v: 0 if int(v) == 0 else 1)
            
            self.labels = torch.Tensor(xy[' Label'])
            drop_cols = [c for c in ['Unnamed: 0', ' Label'] if c in xy.columns]
            xy.drop(drop_cols, axis=1, inplace=True)
        
            print("******************** Train PCA Dataset Loaded **********************")
       
        features = []
        for i in range(n_features):
            features.append('PC '+ str(i+1))
        
        
        xy = xy[features]
        
        self.samples = torch.Tensor(np.array(xy))
        
        del xy
        
        
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, index):      
        
        return self.samples[index], self.labels[index]

    
    
    
class CIC2019Multi(Dataset):
    
    def __init__(self, kind='train', n_features=10):
        if kind=='train':
            print("******************** Loading Train PCA Dataset **********************")
            xy = pd.read_csv(os.getcwd()+'/Datasets/train_PCA_Dataset.csv')    
            # Map labels only if they are strings; if already numeric, keep as-is
            if xy[' Label'].dtype == object:
                xy[' Label'] = xy[' Label'].apply(map_multi_class_attack)
            
            self.labels = torch.Tensor(xy[' Label'])
            drop_cols = [c for c in ['Unnamed: 0', ' Label'] if c in xy.columns]
            xy.drop(drop_cols, axis=1, inplace=True)
            print("******************** Train PCA Dataset Loaded **********************")
    
        else:
            print("******************** Loading Test PCA Dataset **********************")

            xy = pd.read_csv(os.getcwd()+'/Datasets/test_PCA_Dataset.csv')
            if xy[' Label'].dtype == object:
                xy[' Label'] = xy[' Label'].apply(map_multi_class_attack)
            
            self.labels = torch.Tensor(xy[' Label'])
            drop_cols = [c for c in ['Unnamed: 0', ' Label'] if c in xy.columns]
            xy.drop(drop_cols, axis=1, inplace=True)
        
            print("******************** Train PCA Dataset Loaded **********************")
       
        features = []
        for i in range(n_features):
            features.append('PC '+ str(i+1))
        
        
        xy = xy[features]
        
        self.samples = torch.Tensor(np.array(xy))
        
        del xy
        
        
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, index):      
        
        return self.samples[index], self.labels[index]

    

def dataset_loader(BATCH_SIZE):
    """
    This function loads the train and test pca datasets and splits the test set into test and validation
    Return:
        train_loader: train dataset loader
        validation_loader: validatipon dataset loader
        test_loader: test dataset loader
    """
    
    train_data = CIC2019(kind='train')
    test_data = CIC2019(kind='test')
   
    import pdb
    #pdb.set_trace()
    total_size = len(test_data) 
    validation_size = int(0.1 * total_size) 
    test_size = total_size - validation_size    
    test_data, validation_data = random_split(test_data, [test_size, validation_size])
    
    sampler = weighted_random_sampler(train_data)
    

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    
    return train_loader, validation_loader, test_loader


def dataset_loader_multi(BATCH_SIZE):
    """
    This function loads the train and test pca datasets and splits the test set into test and validation
    Return:
        train_loader: train dataset loader
        validation_loader: validatipon dataset loader
        test_loader: test dataset loader
    """
    
    train_data = CIC2019Multi(kind='train')
    test_data = CIC2019Multi(kind='test')
   
    import pdb
    #pdb.set_trace()
    total_size = len(test_data) 
    validation_size = int(0.1 * total_size) 
    test_size = total_size - validation_size    
    test_data, validation_data = random_split(test_data, [test_size, validation_size])
    
    sampler = weighted_random_sampler(train_data)
    

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    
    return train_loader, validation_loader, test_loader

def get_split_dataset(dataset, split=0.2):
    val_size = int(split * len(dataset))
    test_size = len(dataset) - val_size
    return test_size, val_size



def map_multi_class_attack(attack):
    if attack == 'BENIGN':
        attack_type = 0
    elif attack == 'DrDoS_DNS':
        attack_type = 1
    elif attack == 'DrDoS_LDAP':
        attack_type = 2
    elif attack == 'DrDoS_MSSQL':
        attack_type = 3
    elif attack == 'DrDoS_NTP':
        attack_type = 4 
    elif attack == 'DrDoS_NetBIOS':
        attack_type = 5
    elif attack == 'DrDoS_SNMP':
        attack_type = 6
    elif attack == 'DrDoS_SSDP':
        attack_type = 7
    elif attack == 'DrDoS_UDP':
        attack_type = 8
    elif attack == 'Syn':
        attack_type = 9
    elif attack == 'TFTP':
        attack_type = 10
    elif attack == 'UDP-lag':
        attack_type = 11
    elif attack == 'WebDDoS':
        attack_type = 12
    
    return attack_type

def map_binary_class_attack(attack):
    if attack == 'BENIGN':
        attack_type = 0
    else:
        attack_type = 1
    
    return attack_type