"""
Training loop, loss function, and optimizer setup.
"""

import numpy as np
import torch


# ─────────────────────────────────────────────
# Metric-Aligned Loss
# ─────────────────────────────────────────────
def metric_loss(pred, target):
    """
    Spatial RMSE averaged over batch and time.
    pred, target: (B, H, W, T)
    """
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))  # (B, T)
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


# ─────────────────────────────────────────────
# Optimizer & Scheduler
# ─────────────────────────────────────────────
def get_optimizer(cfg, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg['training']['lr'],
        steps_per_epoch=steps_per_epoch,
        epochs=cfg['training']['epochs'],
        pct_start=cfg['training']['pct_start'],
    )
    return optimizer, scheduler


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train(cfg, model, train_dl, val_dl):
    """Full training loop with validation and model checkpointing."""
    device = cfg['device']
    epochs = cfg['training']['epochs']
    grad_clip = cfg['training']['grad_clip']
    save_path = cfg['paths']['model_save']

    optimizer, scheduler = get_optimizer(cfg, model, len(train_dl))

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = metric_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(metric_loss(pred, yb).item())

        train_loss = np.mean(epoch_losses)
        val_loss   = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train RMSE: {train_loss:.4f} | "
                f"Val RMSE: {val_loss:.4f} | "
                f"Best: {best_val_loss:.4f}"
            )

    print(f"\nTraining complete. Best Val RMSE: {best_val_loss:.4f}")
    return history
