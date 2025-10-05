import numpy as np

import torch
import torch.optim as optim
from LIDS.Proposed.model import LCNNModel, LCNNModelMulti
from LIDS.Proposed.data_loader import dataset_loader, dataset_loader_multi
import os
import pandas as pd
from tqdm import tqdm
import sys
from LIDS.eval_tools import accuracy, evaluate_proposed_model
import warnings
from torchviz import make_dot

def _get_current_prefix(default: str = 'CICDDoS2019') -> str:
    try:
        path = os.path.join(os.getcwd(), 'Datasets', 'current_dataset_prefix.txt')
        with open(path, 'r', encoding='utf-8') as f:
            s = f.read().strip()
            return s if s else default
    except Exception:
        return default
#from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"



def trainer(EPOCHS, BATCH_SIZE):
    """
    :param EPOCHS: number of epochs default 1
    :param BATCH_SIZE: batch size default 256
    """
    #writer = SummaryWriter()
    
    
    train_loader, validation_loader, test_loader = dataset_loader(BATCH_SIZE)
    
    # Get the number of features from the first batch
    sample_batch = next(iter(train_loader))
    n_features = sample_batch[0].shape[1]
    print(f"Detected {n_features} PCA features in the dataset")
    
    model = LCNNModel(n_features=n_features).to(device)
    '''
    #Plotting Model
    batch = next(iter(train_loader))
    import pdb
    pdb.set_trace()
    yhat = model(batch[0][0]) # Give dummy batch to forward().
    
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("LCNN", format="png")
    '''
    error = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0


    train_losses = []
    train_acces = []
    
    val_acces = []
    val_losses = []
    
    epoch_train_loss = []
    epoch_train_acc = []
    
    epoch_val_loss = []
    epoch_val_Acc = []

    #early_stopping =
    print('*************** Model Training Started ************** ')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for epoch in range(EPOCHS):
            total_correct = 0
            train_loss = 0

            #loss_idx_value = 0 # tensorboard

            model.train()
            
            
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, (samples, labels) in enumerate(train_bar):
                
                
                # Transfering samples and labels to GPU if available
                samples, labels = samples.to(device), labels.to(device)
                labels = labels.type(torch.int64)

                # Forward pass
                outputs = model(samples)
                loss = error(outputs, labels)

                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Optimizing the parameters
                optimizer.step()

                train_losses.append(loss.item())
                train_acces.append(accuracy(outputs, labels))
            
            model.eval()
            validation_bar = tqdm(validation_loader, file=sys.stdout)

            for step, (samples, labels) in enumerate(validation_bar):

                samples, labels = samples.to(device), labels.to(device)
                outputs = model(samples)
                labels = labels.type(torch.int64)
                loss = error(outputs, labels)
                val_losses.append(loss.item())
                val_acces.append(accuracy(outputs, labels))


           
            #, train_acc, val_loss, val_acc)
            train_loss = round(np.average(train_losses),4)
            train_acc = round(np.average(train_acces), 4)
            val_loss = round(np.average(val_losses), 4)
            val_acc = round(np.average(val_acces), 4)
            

            
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)

            epoch_val_loss.append(val_loss)
            epoch_val_Acc.append(val_acc)
            
            
            
            
            epoch += 1

            # Printing the model Traning Accuracy and Testing Accuracy

            print("Epoch: {}, Train Loss: {}, Val Loss: {}, Train Accuracy: {}, Val Accuracy: {}".format(epoch, train_loss, val_loss, train_acc, val_acc))
        
        train_val_metrics = {'Train Accuracy':epoch_train_acc ,'Train Loss':epoch_train_loss, 'Validation Accuracy':epoch_val_Acc , 'Validation Loss':epoch_val_loss}
        
        train_val_metrics = pd.DataFrame(train_val_metrics)
        prefix = _get_current_prefix()
        out_csv = os.path.join(os.getcwd(), 'Datasets', f'train_val_metrics_{prefix}.csv')
        train_val_metrics.to_csv(out_csv)
       
        print('*************** Model Training Finished ************** ')
        print('*************** Testing Model on the Test Data ************** ')

        
        evaluate_proposed_model(model, test_loader, mode='binary')

        print('*************** Saving the Trained Model ************** ')

        prefix = _get_current_prefix()
        path_to_saved_model = os.path.join(os.getcwd(), 'PretrainedModel')

        # checking if directory exists if not create one
        if not os.path.exists(path_to_saved_model):
            os.mkdir(path_to_saved_model)

        #assigning the model name according to train times with dataset prefix
        version = 1
        while(True):
            name = f'{prefix}_modelv{version}.pth'
            model_name = os.path.join(path_to_saved_model, name)
            if os.path.exists(model_name):
                version += 1
            else:
                break

        torch.save(model.state_dict(), model_name)
        print(f'*************** Model Saved Successfully: {name} ************** ')

        
        

        
def trainer_multi(EPOCHS, BATCH_SIZE):
    """
    :param EPOCHS: number of epochs default 1
    :param BATCH_SIZE: batch size default 1024
    """
    #writer = SummaryWriter()
   
    
    train_loader, validation_loader, test_loader = dataset_loader_multi(BATCH_SIZE)
    
    # Get the number of features from the first batch
    sample_batch = next(iter(train_loader))
    n_features = sample_batch[0].shape[1]
    print(f"Detected {n_features} PCA features in the dataset")
    
    model = LCNNModelMulti(n_features=n_features)
    
    model = model.to(device)
    
   
   
    error = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0


    train_losses = []
    train_acces = []
    
    val_acces = []
    val_losses = []
    
    epoch_train_loss = []
    epoch_train_acc = []
    
    epoch_val_loss = []
    epoch_val_Acc = []

    #early_stopping =
    print('*************** Model Training Started  Multi Class************** ')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for epoch in range(EPOCHS):
            total_correct = 0
            train_loss = 0

            model.train()
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, (samples, labels) in enumerate(train_bar):
                
                # Transfering samples and labels to GPU if available
                samples, labels = samples.to(device), labels.to(device)
                labels = labels.type(torch.int64)
                
                # Forward pass
                outputs = model(samples)
                loss = error(outputs, labels)

                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Optimizing the parameters
                optimizer.step()

                train_losses.append(loss.item())
                train_acces.append(accuracy(outputs, labels))
            
            model.eval()
            validation_bar = tqdm(validation_loader, file=sys.stdout)

            for step, (samples, labels) in enumerate(validation_bar):

                samples, labels = samples.to(device), labels.to(device)
                outputs = model(samples)
                labels = labels.type(torch.int64)
                loss = error(outputs, labels)
                val_losses.append(loss.item())
                val_acces.append(accuracy(outputs, labels))



            #, train_acc, val_loss, val_acc)
            train_loss = round(np.average(train_losses),4)
            train_acc = round(np.average(train_acces), 4)
            val_loss = round(np.average(val_losses), 4)
            val_acc = round(np.average(val_acces), 4)
            

            
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)

            epoch_val_loss.append(val_loss)
            epoch_val_Acc.append(val_acc)
            
            
            
            
            epoch += 1

            # Printing the model Traning Accuracy and Testing Accuracy

            print("Epoch: {}, Train Loss: {}, Val Loss: {}, Train Accuracy: {}, Val Accuracy: {}".format(epoch, train_loss, val_loss, train_acc, val_acc))
        
        train_val_metrics = {'Train Accuracy':epoch_train_acc ,'Train Loss':epoch_train_loss, 'Validation Accuracy':epoch_val_Acc , 'Validation Loss':epoch_val_loss}
        
        train_val_metrics = pd.DataFrame(train_val_metrics)
        prefix = _get_current_prefix()
        out_csv = os.path.join(os.getcwd(), 'Datasets', f'train_val_metrics_multi_{prefix}.csv')
        train_val_metrics.to_csv(out_csv)
       
        print('*************** Model Training Finished ************** ')
        print('*************** Testing Model on the Test Data ************** ')

        
        evaluate_proposed_model(model, test_loader, mode='multi')

        print('*************** Saving the Trained Model ************** ')

        prefix = _get_current_prefix()
        path_to_saved_model = os.path.join(os.getcwd(), 'PretrainedModelMulti')

        # checking if directory exists if not create one
        if not os.path.exists(path_to_saved_model):
            os.mkdir(path_to_saved_model)

        #assigning the model name according to train times with dataset prefix
        version = 1
        while(True):
            name = f'{prefix}_modelMultiv{version}.pth'
            model_name = os.path.join(path_to_saved_model, name)
            if os.path.exists(model_name):
                version += 1
            else:
                break

        torch.save(model.state_dict(), model_name)
        print(f'*************** Model Saved Successfully: {name} ************** ')