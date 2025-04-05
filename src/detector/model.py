import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input"""
    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)

class FeatureAttentionLayer(nn.Module):
    """Graph Feature/Spatial Attention Layer"""
    def __init__(self, n_features, window_size, dropout=0.2, alpha=0.2):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.alpha = alpha
        
        # Feature transformation and attention
        self.fc = nn.Linear(window_size, window_size)
        self.attn = nn.Parameter(torch.empty(size=(2 * window_size, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (batch_size, window_size, n_features)
        x = x.permute(0, 2, 1)  # -> (batch_size, n_features, window_size)
        
        # Compute attention scores
        h = self.fc(x)
        attn_input = torch.cat([x.repeat_interleave(self.n_features, dim=1), 
                              h.repeat(1, self.n_features, 1)], dim=2)
        e = self.leakyrelu(torch.matmul(attn_input, self.attn))
        
        # Apply attention
        attention = torch.softmax(e.view(-1, self.n_features, self.n_features), dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self.sigmoid(torch.bmm(attention, x))
        
        return h.permute(0, 2, 1)  # -> (batch_size, window_size, n_features)

class SupplyChainDetector(nn.Module):
    """Supply Chain Attack Detection Model based on MTAD-GAT"""
    def __init__(self, n_features, window_size, kernel_size=7, dropout=0.2, alpha=0.2):
        super(SupplyChainDetector, self).__init__()
        
        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_attention = FeatureAttentionLayer(n_features, window_size, dropout, alpha)
        
        # GRU layer for temporal dependencies
        self.gru = nn.GRU(input_size=n_features, 
                         hidden_size=64,
                         num_layers=2, 
                         batch_first=True,
                         dropout=dropout)
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Feature extraction
        x = self.conv(x)
        
        # Apply feature attention
        x = self.feature_attention(x)
        
        # Temporal modeling
        x, _ = self.gru(x)
        
        # Get final prediction
        batch_size = x.size(0)
        x = x[:, -1, :]  # Take last timestep
        x = self.fc(x)
        
        return x 