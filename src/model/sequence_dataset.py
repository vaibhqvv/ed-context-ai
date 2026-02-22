import numpy as np, torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class EDSequenceDataset(Dataset):
    """Groups all context objects by (patient_id, stay_id) into variable-length sequences."""

    def __init__(self, context_objects, patient_label_map, device=None):
        # Gather windows per stay, skipping stays with any NaN window
        stays = {}  # (pid, sid) -> [(window_index, vec)]
        labels_map = {}
        for ctx in context_objects:
            label = patient_label_map.get(ctx.patient_id)
            if label is None:
                continue
            vec = ctx.context_vector
            if any(np.isnan(v) for v in vec):
                # mark this stay as tainted
                stays.pop((ctx.patient_id, ctx.stay_id), None)
                labels_map.pop((ctx.patient_id, ctx.stay_id), None)
                continue
            key = (ctx.patient_id, ctx.stay_id)
            if key not in stays:
                stays[key] = []
                labels_map[key] = label
            stays[key].append((ctx.window_index, vec))

        # Sort windows within each stay and build tensors
        self.sequences = []  # list of (seq_len, feat_dim) tensors
        self.labels = []
        self.patient_ids = []
        for (pid, _sid), windows in stays.items():
            windows.sort(key=lambda w: w[0])
            seq = torch.tensor([w[1] for w in windows], dtype=torch.float32)
            self.sequences.append(seq)
            self.labels.append(float(labels_map[(pid, _sid)]))
            self.patient_ids.append(pid)

        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        if device is not None:
            self.sequences = [s.to(device) for s in self.sequences]
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def sequence_collate_fn(batch):
    """Collate variable-length sequences into a padded batch.

    Returns:
        padded_seqs: (batch, max_seq_len, feat_dim)
        labels: (batch,)
        lengths: (batch,) int64 actual sequence lengths
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded, labels, lengths
