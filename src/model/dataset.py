import pickle, numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.config_loader import get_config

cfg = get_config()


class EDContextDataset(Dataset):
    def __init__(self, context_objects, patient_label_map, device=None):
        # Group by (patient_id, stay_id) and keep only last window per stay
        best = {}  # (patient_id, stay_id) -> (window_index, vec, label)
        for ctx in context_objects:
            label = patient_label_map.get(ctx.patient_id)
            if label is None:
                continue
            vec = ctx.context_vector
            if any(np.isnan(v) for v in vec):
                continue
            key = (ctx.patient_id, ctx.stay_id)
            if key not in best or ctx.window_index > best[key][0]:
                best[key] = (ctx.window_index, vec, label)

        vecs, labels, patient_ids, stay_ids = [], [], [], []
        for (pid, sid), (_widx, vec, label) in best.items():
            vecs.append(vec)
            labels.append(label)
            patient_ids.append(pid)
            stay_ids.append(sid)

        self.X = torch.tensor(vecs, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.patient_ids = patient_ids
        self.stay_ids = stay_ids
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_label_map():
    pkl_path = Path(cfg["paths"]["processed_data"]) / "patient_records.pkl"
    label_map = {}
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    for r in records:
        label_map[r.patient_id] = r.outcome_binary
    del records
    return label_map
