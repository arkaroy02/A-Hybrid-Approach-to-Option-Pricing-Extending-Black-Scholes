import torch
from torch import nn

class HybridNetV3(nn.Module):
    def __init__(self, input_dim, sigma_index):
        super().__init__()
        
        self.sigma_index = sigma_index
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.log_error_head = nn.Linear(64, 1)
        self.alpha_head = nn.Linear(64, 1)
        
    def forward(self, x, bs):
        h = self.shared(x)
        
        log_error = self.log_error_head(h).squeeze()

        # 🔥 volatility-aware damping
        sigma = x[:, self.sigma_index]
        sigma_scaled = torch.sigmoid(sigma)
        log_error = log_error / (1 + 0.5 * sigma_scaled)

        alpha_raw = self.alpha_head(h).squeeze()
        alpha = torch.sigmoid(alpha_raw - 0.5 * log_error)

        ml_price = bs * torch.exp(log_error)
        hybrid = alpha * bs + (1 - alpha) * ml_price

        return hybrid, alpha, log_error