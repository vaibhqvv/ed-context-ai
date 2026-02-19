import pickle, numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.config_loader import get_config

cfg = get_config()


class EDContextDataset(Dataset):
    def __init__(self, context_objects, patient_label_map):
        self.samples = []
        for ctx in context_objects:
            label = patient_label_map.get(ctx.patient_id)
            if label is None:
                continue
            if any(np.isnan(v) for v in ctx.context_vector):
                continue
            self.samples.append((ctx.context_vector, float(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vec, label = self.samples[idx]
        return torch.tensor(vec, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


def load_label_map():
    with open(Path(cfg["paths"]["processed_data"]) / "patient_records.pkl", "rb") as f:
        records = pickle.load(f)
    return {r.patient_id: r.outcome_binary for r in records}
