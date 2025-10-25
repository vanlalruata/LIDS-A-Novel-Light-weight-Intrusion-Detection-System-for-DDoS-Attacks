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

# Optional plotting inside SHAP helpers
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
                
                
                # Transferring samples and labels to GPU if available
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

            # model.summary()

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

            # Printing the model Training Accuracy and Testing Accuracy

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
    Multi-class trainer. For CICDDoS2019, use improved CNN + class weights + early stopping + SHAP.
    For other datasets, keep existing behavior.
    """
    #writer = SummaryWriter()
    
    prefix = _get_current_prefix()
    train_loader, validation_loader, test_loader = dataset_loader_multi(BATCH_SIZE)
    
    # Get the number of features from the first batch
    sample_batch = next(iter(train_loader))
    n_features = sample_batch[0].shape[1]
    print(f"Detected {n_features} PCA features in the dataset")

    is_cic = str(prefix).lower().startswith('cic')

    # --- Helpers from CNN_model.py (minimal inline copy) ---
    class EarlyStopping:
        def __init__(self, patience=8, min_delta=1e-4, mode="min"):
            self.patience = patience
            self.min_delta = min_delta
            self.mode = mode
            self.best = None
            self.counter = 0
            self.best_state = None
        def step(self, metric, model):
            improve = False
            if self.best is None:
                improve = True
            else:
                if self.mode == "min":
                    improve = (self.best - metric) > self.min_delta
                else:
                    improve = (metric - self.best) > self.min_delta
            if improve:
                self.best = metric
                self.counter = 0
                self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                self.counter += 1
            return self.counter >= self.patience

    def compute_class_weights(loader, num_classes):
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, y in loader:
            y = y.view(-1).cpu().numpy().astype(int)
            for v in y:
                if 0 <= v < num_classes:
                    counts[v] += 1
        counts = np.clip(counts, 1, None)
        inv = 1.0 / counts.astype(float)
        weights = inv / inv.sum() * len(inv)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def run_shap_kernel_explainer_per_class(model, test_loader, out_dir="Images/shap", nsamples=200, cap_background=200, cap_explain=800):
        try:
            import shap  # noqa
        except Exception as e:
            print(f"[SHAP] shap not available: {e}")
            return
        os.makedirs(out_dir, exist_ok=True)
        Xtest = []
        with torch.no_grad():
            for X, _ in test_loader:
                Xtest.append(X.cpu().numpy())
                if sum(arr.shape[0] for arr in Xtest) >= max(cap_explain, cap_background):
                    break
        if not Xtest:
            print("[SHAP] No test samples collected.")
            return
        X_test_np = np.concatenate(Xtest, axis=0)
        N, D = X_test_np.shape
        with torch.no_grad():
            tmp_logits = model(torch.tensor(X_test_np[:4], dtype=torch.float32, device=device))
            C = tmp_logits.shape[1]
        feature_names = [f"PC {i+1}" for i in range(D)]
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(N, size=min(cap_background, N), replace=False)
        ex_idx = rng.choice(N, size=min(cap_explain, N), replace=False)
        X_bg = X_test_np[bg_idx]
        X_ex = X_test_np[ex_idx]
        for c in range(C):
            def prob_c(Xnp):
                Xt = torch.tensor(Xnp, dtype=torch.float32, device=device)
                with torch.no_grad():
                    probs = torch.softmax(model(Xt), dim=1)[:, c]
                return probs.detach().cpu().numpy()
            explainer_c = shap.KernelExplainer(prob_c, X_bg)
            sv_c = explainer_c.shap_values(X_ex, nsamples=nsamples)
            if isinstance(sv_c, list):
                sv_c = sv_c[0]
            if not isinstance(sv_c, np.ndarray) or sv_c.ndim != 2 or sv_c.shape[1] != D:
                print(f"[SHAP] Unexpected sv shape for class {c}: {getattr(sv_c,'shape',None)}")
                continue
            mean_abs = np.abs(sv_c).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:min(D, 10)]
            if plt is not None:
                plt.figure(figsize=(6.2, 3.6))
                plt.barh(range(len(order)), mean_abs[order])
                plt.gca().invert_yaxis()
                plt.yticks(range(len(order)), [feature_names[i] for i in order])
                plt.xlabel("Mean |SHAP|")
                plt.title(f"Top features for class {c}")
                plt.tight_layout()
                out_png = os.path.join(out_dir, f"shap_top_features_class_{c}.png")
                plt.savefig(out_png, dpi=300)
                plt.close()
                print(f"[SHAP] Saved → {out_png}")

    if is_cic:
        # Determine number of classes from a pass over training loader
        with torch.no_grad():
            ys = []
            for _, y in train_loader:
                ys.append(y.view(-1).cpu().numpy())
                if len(ys) > 10:
                    break
            ycat = np.unique(np.concatenate(ys).astype(int))
            num_classes = int(ycat.max()) + 1
        print(f"[CIC] Detected {num_classes} classes")

        class CICLCNNModelMulti(torch.nn.Module):
            def __init__(self, num_classes, n_features):
                super().__init__()
                self.conv = torch.nn.Conv1d(1, 16, kernel_size=1, padding=0)
                self.bn = torch.nn.BatchNorm1d(16)
                self.relu = torch.nn.ReLU(inplace=True)
                self.pool = torch.nn.MaxPool1d(kernel_size=2)  # D -> D//2
                self.fc1 = torch.nn.Linear(16 * (n_features // 2), 64)
                self.drop = torch.nn.Dropout(p=0.4)
                self.fc2 = torch.nn.Linear(64, num_classes)
            def forward(self, x):
                x = x.unsqueeze(1)
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.drop(x)
                return self.fc2(x)

        model = CICLCNNModelMulti(num_classes=num_classes, n_features=n_features).to(device)
        # Class weights for imbalance
        class_weights = compute_class_weights(train_loader, num_classes)
        error = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        early = EarlyStopping(patience=8, min_delta=1e-4, mode="min")
    
    else:
        # Default behavior for other datasets
        model = LCNNModelMulti(n_features=n_features).to(device)
        error = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early = None

    epoch = 0

    train_losses = []
    train_acces = []
    val_acces = []
    val_losses = []
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_Acc = []

    print('*************** Model Training Started  Multi Class************** ')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(EPOCHS):
            model.train()
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, (samples, labels) in enumerate(train_bar):
                samples, labels = samples.to(device), labels.to(device)
                labels = labels.type(torch.int64)
                outputs = model(samples)
                loss = error(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
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

            train_loss = round(np.average(train_losses), 4)
            train_acc = round(np.average(train_acces), 4)
            val_loss = round(np.average(val_losses), 4)
            val_acc = round(np.average(val_acces), 4)

            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
            epoch_val_loss.append(val_loss)
            epoch_val_Acc.append(val_acc)
            epoch += 1

            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train Accuracy: {train_acc}, Val Accuracy: {val_acc}")

            # Early stopping on val loss (CIC only)
            if early is not None:
                if early.step(val_loss, model):
                    print(f"[EarlyStopping] Patience reached at epoch {epoch}. Restoring best weights.")
                    break

        # Restore best weights if early stopping used
        if early is not None and early.best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in early.best_state.items()})
            print(f"Best val loss: {early.best:.6f}")

        train_val_metrics = {'Train Accuracy': epoch_train_acc, 'Train Loss': epoch_train_loss,
                             'Validation Accuracy': epoch_val_Acc, 'Validation Loss': epoch_val_loss}
        train_val_metrics = pd.DataFrame(train_val_metrics)
        prefix = _get_current_prefix()
        out_csv = os.path.join(os.getcwd(), 'Datasets', f'train_val_metrics_multi_{prefix}.csv')
        train_val_metrics.to_csv(out_csv)

        print('*************** Model Training Finished ************** ')
        print('*************** Testing Model on the Test Data ************** ')
        evaluate_proposed_model(model, test_loader, mode='multi')

        # SHAP only for CIC
        if is_cic:
            try:
                run_shap_kernel_explainer_per_class(model, test_loader, out_dir=os.path.join('Images', 'shap'))
            except Exception as e:
                print(f"[SHAP] Failed to run SHAP: {e}")

        print('*************** Saving the Trained Model ************** ')
        prefix = _get_current_prefix()
        path_to_saved_model = os.path.join(os.getcwd(), 'PretrainedModelMulti')
        if not os.path.exists(path_to_saved_model):
            os.mkdir(path_to_saved_model)
        version = 1
        while True:
            name = f'{prefix}_modelMultiv{version}.pth'
            model_name = os.path.join(path_to_saved_model, name)
            if os.path.exists(model_name):
                version += 1
            else:
                break
        torch.save(model.state_dict(), model_name)
        print(f'*************** Model Saved Successfully: {name} ************** ')