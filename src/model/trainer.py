import torch, numpy as np, pickle
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from src.model.mc_dropout import MCDropoutNet
from src.model.dataset import EDContextDataset, load_label_map
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = get_config()


def train():
    log.info("=== Training MC Dropout Model ===")
    with open(Path(cfg["paths"]["context_data"]) / "context_objects.pkl", "rb") as f:
        contexts = pickle.load(f)
    label_map = load_label_map()
    dataset = EDContextDataset(contexts, label_map)
    log.info(f"Dataset: {len(dataset)} samples")

    x0, _ = dataset[0]
    input_dim = len(x0)
    cfg["model"]["input_dim"] = input_dim
    log.info(f"Input dim: {input_dim}")
    val_size = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["model"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=256)

    model = MCDropoutNet(input_dim=input_dim)
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
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for X, y in val_loader:
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
            torch.save(
                {"model_state": model.state_dict(), "input_dim": input_dim},
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
