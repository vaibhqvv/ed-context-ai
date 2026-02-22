import gc, os
import torch, numpy as np, pickle
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from src.model.mc_dropout import MCDropoutNet
from src.model.mc_dropout_gru import MCDropoutGRU
from src.model.dataset import EDContextDataset, load_label_map
from src.model.sequence_dataset import EDSequenceDataset, sequence_collate_fn
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.device import get_device

log = get_logger(__name__)
cfg = get_config()


def _load_contexts():
    """Load raw context objects from pickle with progress bar."""
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
    return contexts


def train():
    model_type = cfg["model"].get("model_type", "mlp")
    log.info(f"=== Training MC Dropout Model ({model_type.upper()}) ===")
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    contexts = _load_contexts()
    label_map = load_label_map()

    # --- Build dataset based on model type ---
    if model_type == "gru":
        dataset = EDSequenceDataset(contexts, label_map, device=device)
    else:
        dataset = EDContextDataset(contexts, label_map, device=device)

    del contexts, label_map
    gc.collect()

    log.info(f"Dataset: {len(dataset)} samples")

    # --- Determine input_dim ---
    if model_type == "gru":
        input_dim = dataset.sequences[0].shape[-1]
    else:
        input_dim = dataset.X.shape[1]
    cfg["model"]["input_dim"] = input_dim
    log.info(f"Input dim: {input_dim}")

    # --- Patient-level split ---
    unique_pids = list(set(dataset.patient_ids))
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    val_count = int(0.2 * len(unique_pids))
    val_pids = set(unique_pids[:val_count])
    train_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid not in val_pids]
    val_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_pids]
    log.info(f"Patient-level split: {len(unique_pids)-val_count} train / {val_count} val patients")

    # --- Standardize ---
    if model_type == "gru":
        # Concatenate all training sequences to compute mean/std
        train_vecs = torch.cat([dataset.sequences[i] for i in train_idx], dim=0)
        mean = train_vecs.mean(dim=0)
        std = train_vecs.std(dim=0)
        std[std < 1e-8] = 1.0
        del train_vecs
        dataset.sequences = [(s - mean) / std for s in dataset.sequences]
    else:
        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=dataset.X.device)
        mean = dataset.X[train_idx_t].mean(dim=0)
        std = dataset.X[train_idx_t].std(dim=0)
        std[std < 1e-8] = 1.0
        del train_idx_t
        dataset.X = (dataset.X - mean) / std
    log.info("Standardized input features (train stats only)")

    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)

    # --- DataLoaders ---
    on_gpu = device.type == "cuda"
    n_workers = 0 if on_gpu else min(os.cpu_count() or 1, 2)
    log.info(f"DataLoader workers: {n_workers} (data on {'GPU' if on_gpu else 'CPU'})")
    batch_size = cfg["model"]["batch_size"]

    loader_kwargs = {}
    if model_type == "gru":
        loader_kwargs["collate_fn"] = sequence_collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2,
        num_workers=n_workers, **loader_kwargs,
    )

    # --- Model ---
    if model_type == "gru":
        model = MCDropoutGRU(input_dim=input_dim).to(device)
    else:
        model = MCDropoutNet(input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["model"]["learning_rate"], weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["model"]["max_epochs"], eta_min=1e-5
    )

    # --- Class balance ---
    if model_type == "gru":
        train_labels = torch.tensor([dataset.labels[i] for i in train_idx])
    else:
        train_labels = dataset.y[train_idx]
    n_pos = int(train_labels.sum().item())
    n_neg = len(train_idx) - n_pos
    log.info(f"Train class balance: pos={n_pos}, neg={n_neg}")
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    log.info(f"pos_weight: {pos_weight.item():.2f}")
    train_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_criterion = torch.nn.BCEWithLogitsLoss()

    best_auroc = 0.0
    patience_ctr = 0
    patience = cfg["model"]["early_stopping_patience"]
    save_path = Path(cfg["paths"]["model_output"])
    save_path.mkdir(parents=True, exist_ok=True)

    is_gru = model_type == "gru"

    for epoch in range(cfg["model"]["max_epochs"]):
        model.train()
        t_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for batch in pbar:
            if is_gru:
                X, y, lengths = batch
                X, y, lengths = X.to(device), y.to(device), lengths
            else:
                X, y = batch
                lengths = None
            optimizer.zero_grad()
            logits = model(X, lengths) if is_gru else model(X)
            loss = train_criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        v_losses, all_logits, all_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:3d} [val]", leave=False):
                if is_gru:
                    X, y, lengths = batch
                    X, y, lengths = X.to(device), y.to(device), lengths
                    logits = model(X, lengths)
                else:
                    X, y = batch
                    logits = model(X)
                v_losses.append(val_criterion(logits, y).item())
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        v_loss = np.mean(v_losses)
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
        val_auroc = roc_auc_score(all_labels, val_probs)
        val_auprc = average_precision_score(all_labels, val_probs)

        scheduler.step()
        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:3d} | Train(w): {np.mean(t_losses):.4f} | Val: {v_loss:.4f}"
                f" | AUROC: {val_auroc:.3f} | AUPRC: {val_auprc:.3f}"
            )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_ctr = 0
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": cpu_state,
                    "model_type": model_type,
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

    log.info(f"Training done. Best val AUROC: {best_auroc:.4f}")
    return save_path / "best_model.pt"


if __name__ == "__main__":
    train()
