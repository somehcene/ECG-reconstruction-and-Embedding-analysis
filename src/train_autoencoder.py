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


def compute_lead_reconstruction_matrix(model, loader, device="cuda", n_samples=None):
    """
    Computes a 12x12 reconstruction matrix M[i, j] where:
        M[i, j] = mean MSE of reconstructing lead j when lead i is used as input.

    The diagonal M[i, i] is set to NaN (trivial same-lead reconstruction).

    Args:
        model      : trained UNet1D model
        loader     : DataLoader (test set)
        device     : computation device
        n_samples  : if not None, stop after processing approximately n_samples ECGs

    Returns:
        mse_matrix  : (12, 12) numpy array, NaN on diagonal
        corr_matrix : (12, 12) numpy array, NaN on diagonal
    """
    import numpy as np

    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_leads = 12
    mse_sums  = np.zeros((n_leads, n_leads))
    corr_sums = np.zeros((n_leads, n_leads))
    counts    = np.zeros((n_leads, n_leads))

    n_processed = 0

    with torch.no_grad():
        for signals, _ in loader:

            if n_samples is not None and n_processed >= n_samples:
                break

            signals    = signals.to(device)   # (B, 12, T)
            batch_size = signals.shape[0]

            for i in range(n_leads):

                input_lead = signals[:, i:i+1, :]  # (B, 1, T)
                pred, _    = model(input_lead)       # pred: (B, 12, T)
                target     = signals                  # (B, 12, T)

                # MSE per output lead j  →  shape (12,)
                mse_per_lead = ((pred - target) ** 2).mean(dim=(0, 2))

                # Pearson correlation per output lead j  →  shape (B, 12) → mean (12,)
                x = pred   - pred.mean(dim=-1, keepdim=True)
                y = target - target.mean(dim=-1, keepdim=True)
                corr_per_lead = (x * y).sum(dim=-1) / (
                    torch.sqrt((x ** 2).sum(dim=-1) * (y ** 2).sum(dim=-1) + 1e-8)
                )
                corr_per_lead = corr_per_lead.mean(dim=0)  # (12,)

                for j in range(n_leads):
                    mse_sums[i, j]  += mse_per_lead[j].item()
                    corr_sums[i, j] += corr_per_lead[j].item()
                    counts[i, j]    += 1

            n_processed += batch_size

    mse_matrix  = mse_sums  / counts
    corr_matrix = corr_sums / counts

    # Mask diagonal (same-lead reconstruction is trivial)
    np.fill_diagonal(mse_matrix,  np.nan)
    np.fill_diagonal(corr_matrix, np.nan)

    print("\n===== LEAD RECONSTRUCTION MATRIX =====")
    print(f"Mean MSE  (off-diagonal): {np.nanmean(mse_matrix):.4f}")
    print(f"Mean Corr (off-diagonal): {np.nanmean(corr_matrix):.4f}")

    return mse_matrix, corr_matrix