import torch

def custom_loss(hybrid, market, alpha, log_error, bs,
                lambda_alpha=0.05,
                lambda_dir=0.15,
                lambda_ml=0.01):

    # Base error
    mse = torch.mean((hybrid - market) ** 2)
    
    # Reduce alpha randomness
    alpha_reg = torch.var(alpha)
    
    # Directional penalty
    gap = (market - bs) / (bs + 1e-6)
    error = hybrid - market

    dir_penalty = torch.mean(
        (gap > 0).float() * torch.relu(-error) * 2.5 +   
        (gap < 0).float() * torch.relu(error) * 2
    )
    
    # ML stability
    ml_reg = torch.mean(log_error ** 2)

    return mse + lambda_alpha * alpha_reg + lambda_dir * dir_penalty + lambda_ml * ml_reg