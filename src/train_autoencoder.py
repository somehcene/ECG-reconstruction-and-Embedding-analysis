import torch
from tqdm import tqdm



# LOSS (MSE)


def reconstruction_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)



# TRAIN ONE EPOCH


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for signals, _ in tqdm(loader, desc="Training AE"):

        signals = signals.to(device)  # (B, 12, T)

        # input = 1 lead
        lead_idx = torch.randint(0, 12, (signals.shape[0],), device=signals.device)

        input_lead = torch.stack([
            signals[i, idx:idx+1, :]
            for i, idx in enumerate(lead_idx)
        ])
        target = signals                 # 12 leads

        optimizer.zero_grad()

        pred, _ = model(input_lead)

        loss = reconstruction_loss(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# EVALUATION


def evaluate(model, loader, device):
    model.eval()

    total_mse = 0
    total_corr = 0
    n = 0

    with torch.no_grad():
        for signals, _ in tqdm(loader, desc="Evaluating AE"):

            signals = signals.to(device)

            input_lead = signals[:, 0:1, :]
            target = signals

            pred, _ = model(input_lead)

            # MSE
            mse = torch.mean((pred - target) ** 2)

            # Pearson
            x = pred - pred.mean(dim=-1, keepdim=True)
            y = target - target.mean(dim=-1, keepdim=True)

            corr = torch.sum(x * y, dim=-1) / (
                torch.sqrt(torch.sum(x**2, dim=-1) * torch.sum(y**2, dim=-1) + 1e-8)
            )

            total_mse += mse.item()
            total_corr += corr.mean().item()
            n += 1

    return total_mse / n, total_corr / n



# TRAIN LOOP


def train_autoencoder(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mse, val_corr = evaluate(model, val_loader, device)

        print(f"Train Loss (MSE): {train_loss:.4f}")
        print(f"Val MSE: {val_mse:.4f}")
        print(f"Val Corr: {val_corr:.4f}")


def dtw_distance(x, y):
    """
    Simple DTW implementation (CPU, numpy-like)
    x, y: (T,)
    """
    import numpy as np

    x = x.flatten()
    y = y.flatten()

    n, m = len(x), len(y)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            )

    return dtw[n, m]


def test_autoencoder(model, test_loader, device="cuda"):
    model.eval()

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_mse = 0
    total_corr = 0
    total_dtw = 0
    n = 0

    with torch.no_grad():
        for signals, _ in test_loader:

            signals = signals.to(device)

            lead_idx = torch.randint(0, 12, (signals.shape[0],), device=signals.device)

            input_lead = torch.stack([
                signals[i, idx:idx+1, :]
                for i, idx in enumerate(lead_idx)
            ])
            target = signals

            pred, _ = model(input_lead)

            
            # MSE
            
            mse = torch.mean((pred - target) ** 2)

            
            # Pearson
            
            x = pred - pred.mean(dim=-1, keepdim=True)
            y = target - target.mean(dim=-1, keepdim=True)

            corr = torch.sum(x * y, dim=-1) / (
                torch.sqrt(torch.sum(x**2, dim=-1) * torch.sum(y**2, dim=-1) + 1e-8)
            )

            
            # DTW (CPU)
            
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            batch_dtw = 0

            for b in range(pred_np.shape[0]):
                for lead in range(pred_np.shape[1]):
                    batch_dtw += dtw_distance(
                        pred_np[b, lead],
                        target_np[b, lead]
                    )

            batch_dtw /= (pred_np.shape[0] * pred_np.shape[1])

            
            # Accumulate
            
            total_mse += mse.item()
            total_corr += corr.mean().item()
            total_dtw += batch_dtw
            n += 1

    final_mse = total_mse / n
    final_rmse = final_mse ** 0.5
    final_corr = total_corr / n
    final_dtw = total_dtw / n

    print("\n===== TEST RESULTS =====")
    print(f"MSE:   {final_mse:.4f}")
    print(f"RMSE:  {final_rmse:.4f}")
    print(f"Corr:  {final_corr:.4f}")
    print(f"DTW:   {final_dtw:.4f}")

    return final_mse, final_rmse, final_corr, final_dtw