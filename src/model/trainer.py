import gc, os
import torch, numpy as np, pickle
from pathlib import Path
from torch.utils.data import DataLoader, Subset
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+ GPUs

    # --- Load data and build dataset, then free raw objects ---
    pkl_path = Path(cfg["paths"]["context_data"]) / "context_objects.pkl"
    file_size = pkl_path.stat().st_size
    log.info(f"Loading context objects ({file_size / 1e6:.0f} MB)...")
    with open(pkl_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Loading pkl") as pbar:
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
    dataset = EDContextDataset(contexts, label_map, device=device)

    # Free the raw context objects and label map — dataset now holds compact tensors
    del contexts, label_map
    gc.collect()

    log.info(f"Dataset: {len(dataset)} samples")

    input_dim = dataset.X.shape[1]
    cfg["model"]["input_dim"] = input_dim
    log.info(f"Input dim: {input_dim}")

    # Patient-level split to prevent data leakage
    unique_pids = list(set(dataset.patient_ids))
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    val_count = int(0.2 * len(unique_pids))
    val_pids = set(unique_pids[:val_count])
    train_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid not in val_pids]
    val_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_pids]
    log.info(f"Patient-level split: {len(unique_pids)-val_count} train / {val_count} val patients")

    # Standardize using train statistics only (no val leakage)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=dataset.X.device)
    mean = dataset.X[train_idx_t].mean(dim=0)
    std = dataset.X[train_idx_t].std(dim=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant features
    del train_idx_t
    dataset.X = (dataset.X - mean) / std
    log.info("Standardized input features (train stats only)")

    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)

    # Data is already on GPU — no need for workers or pin_memory
    on_gpu = device.type == "cuda"
    n_workers = 0 if on_gpu else min(os.cpu_count() or 1, 2)
    log.info(f"DataLoader workers: {n_workers} (data on {'GPU' if on_gpu else 'CPU'})")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["model"]["batch_size"],
        shuffle=True,
        num_workers=n_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["model"]["batch_size"] * 2,
        num_workers=n_workers,
    )

    model = MCDropoutNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["model"]["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    # Compute class weight from training set
    train_labels = dataset.y[train_idx]
    n_pos = int(train_labels.sum().item())
    n_neg = len(train_idx) - n_pos
    log.info(f"Train class balance: pos={n_pos}, neg={n_neg}")
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    log.info(f"pos_weight: {pos_weight.item():.2f}")
    train_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_criterion = torch.nn.BCEWithLogitsLoss()  # unweighted for true val quality

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
            optimizer.zero_grad()
            loss = train_criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        v_losses = []
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch:3d} [val]", leave=False):
                v_losses.append(val_criterion(model(X), y).item())

        v_loss = np.mean(v_losses)
        scheduler.step(v_loss)
        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:3d} | Train(w): {np.mean(t_losses):.4f} | Val: {v_loss:.4f}"
            )

        if v_loss < best_loss:
            best_loss = v_loss
            patience_ctr = 0
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": cpu_state,
                    "input_dim": input_dim,
                    "feature_mean": mean.cpu(),
                    "feature_std": std.cpu(),
                },
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
