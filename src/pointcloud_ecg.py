import torch


def create_point_cloud(signals, embeddings):
    """
    signals: (B, 12, T)
    embeddings: (B, 12, d)

    returns:
        point_cloud: (B, 12*T, 3 + d)
    """

    B, L, T = signals.shape
    d = embeddings.shape[2]

    device = signals.device

    # create time and lead grids
    time = torch.arange(T, device=device).float()
    lead = torch.arange(L, device=device).float()

    time_grid = time.view(1, 1, T).expand(B, L, T)
    lead_grid = lead.view(1, L, 1).expand(B, L, T)

    # voltage
    voltage = signals  # (B, L, T)

    # expand embeddings across time
    emb = embeddings.unsqueeze(-1).expand(B, L, d, T)  # (B, L, d, T)

    # reshape everything
    time_grid = time_grid.reshape(B, L * T, 1)
    lead_grid = lead_grid.reshape(B, L * T, 1)
    voltage = voltage.reshape(B, L * T, 1)

    emb = emb.permute(0, 1, 3, 2).reshape(B, L * T, d)

    # concat → final point cloud
    point_cloud = torch.cat([time_grid, lead_grid, voltage, emb], dim=-1)

    return point_cloud