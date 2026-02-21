import os
import torch, numpy as np, pickle
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.model.mc_dropout import MCDropoutNet
from src.model.dataset import EDContextDataset, load_label_map
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.device import get_device

log = get_logger(__name__)
cfg = get_config()


def train():
    log.info("=== Training MC Dropout Model ===")
    device = get_device()
    pkl_path = Path(cfg["paths"]["context_data"]) / "context_objects.pkl"
    file_size = pkl_path.stat().st_size
    log.info(f"Loading context objects ({file_size / 1e6:.0f} MB)...")
    with open(pkl_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Loading pkl") as pbar:
            # Wrap file so tqdm tracks bytes read by pickle
            class _TrackedReader:
                def __init__(self, fh, pbar):
                    self._fh, self._pbar = fh, pbar
                def read(self, n=-1):
                    data = self._fh.read(n)
                    self._pbar.update(len(data))
                    return data
                def readline(self):
                    line = self._fh.readline()
                    self._pbar.update(len(line))
                    return line
            contexts = pickle.load(_TrackedReader(f, pbar))
    label_map = load_label_map()
    dataset = EDContextDataset(contexts, label_map)
    log.info(f"Dataset: {len(dataset)} samples")

    x0, _ = dataset[0]
    input_dim = len(x0)
    cfg["model"]["input_dim"] = input_dim
    log.info(f"Input dim: {input_dim}")
    val_size = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    n_workers = min(os.cpu_count() or 1, 8)
    log.info(f"DataLoader workers: {n_workers}")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["model"]["batch_size"],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )

    model = MCDropoutNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["model"]["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = torch.nn.BCELoss()

    # Compute class weight for imbalanced labels
    labels = [s[1] for s in dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    log.info(f"Class balance: pos={int(n_pos)}, neg={int(n_neg)}")

    best_loss = float("inf")
    patience_ctr = 0
    patience = cfg["model"]["early_stopping_patience"]
    save_path = Path(cfg["paths"]["model_output"])
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["model"]["max_epochs"]):
        model.train()
        t_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for X, y in pbar:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        v_losses = []
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch:3d} [val]", leave=False):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                v_losses.append(criterion(model(X), y).item())

        v_loss = np.mean(v_losses)
        scheduler.step(v_loss)
        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:3d} | Train: {np.mean(t_losses):.4f} | Val: {v_loss:.4f}"
            )

        if v_loss < best_loss:
            best_loss = v_loss
            patience_ctr = 0
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"model_state": cpu_state, "input_dim": input_dim},
                save_path / "best_model.pt",
            )
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"Early stop at epoch {epoch}")
                break

    log.info(f"Training done. Best val loss: {best_loss:.4f}")
    return save_path / "best_model.pt"


if __name__ == "__main__":
    train()
