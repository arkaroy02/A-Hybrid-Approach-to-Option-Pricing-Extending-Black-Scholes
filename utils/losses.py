import torch

def custom_loss(
    hybrid,
    market,
    alpha,
    log_error,
    bs,
    lambda_alpha=0.05,
    lambda_dir=0.3,
    lambda_ml=0.01,
):
    mse = torch.mean((hybrid - market) ** 2)
    alpha_reg = torch.var(alpha)

    direction = torch.sign(market - bs)
    model_dir = torch.sign(hybrid - bs)

    wrong_direction = (direction != model_dir).float()
    dir_penalty = torch.mean(wrong_direction * torch.abs(hybrid - market))
    ml_reg = torch.mean(log_error ** 2)

    return mse + lambda_alpha * alpha_reg + lambda_dir * dir_penalty + lambda_ml * ml_reg
