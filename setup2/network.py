import torch
import torch.nn as nn
import torch.nn.functional as F



class Lstm(nn.Module):

    def __init__(self,n_features=5,n_classes=4,n_hidden=256,n_layers=2,dropout=0.3):
        super().__init__()
        self.n_hidden=n_hidden
        self.n_layers=n_layers
        self.n_classes=n_classes
        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            bidirectional=True,
            batch_first = True,
            dropout=dropout
        )    
        
        self.fc = nn.Linear(n_hidden*2, n_classes)
  

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out, (hn, cn)
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.n_hidden)
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.n_hidden)
    
        # Determine the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Move tensors to the determined device
        h0, c0 = h0.to(device), c0.to(device)
    
        return h0, c0
