from torch import nn


class LCNNModel(nn.Module):

    def __init__(self, n_features=10):
        super().__init__()
        
        self.n_features = n_features

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, padding=1),
            # nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # We'll calculate the actual size dynamically in forward pass
        # For now, create a dummy layer that will be replaced
        self.fc1 = None
        self.drop = nn.Dropout2d(0.6)
        self.fc2 = nn.Linear(in_features=64, out_features=2)
        
        self._initialized = False

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)

        out = out.view(out.size(0), -1)
        
        # Initialize fc1 on first forward pass with actual dimensions
        if not self._initialized:
            actual_size = out.shape[1]
            self.fc1 = nn.Linear(in_features=actual_size, out_features=64).to(out.device)
            self._initialized = True
            print(f"Initialized fc1 with input size: {actual_size}")
        
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


class LCNNModelMulti(nn.Module):

    def __init__(self, n_features=10):
        super().__init__()
        
        self.n_features = n_features

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1, padding=1),
            # nn.BatchNorm2d(20),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, padding=1),
            # nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # We'll calculate the actual size dynamically in forward pass
        self.fc1 = None
        self.drop = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=13)
        
        self._initialized = False

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        #print(out.shape)
       
        out = self.layer2(out)
        #print(out.shape)

        out = out.view(out.size(0), -1)
        
        # Initialize fc1 on first forward pass with actual dimensions
        if not self._initialized:
            actual_size = out.shape[1]
            self.fc1 = nn.Linear(in_features=actual_size, out_features=128).to(out.device)
            self._initialized = True
            print(f"Initialized fc1 with input size: {actual_size}")
        
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out
