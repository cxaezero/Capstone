import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset

class NPYPairedDataset(Dataset):
    def __init__(self, list_path, root=None, n_len=800, test_mode=False):
        self.test_mode = test_mode
        self.root = root
        self.file_list = [line.strip() for line in open(list_path) if line.strip()]
        self.n_len = n_len

        if not test_mode:
            self.a_len = len(self.file_list) - self.n_len
            self.normal_indices = list(range(self.a_len, len(self.file_list)))
            self.abnormal_indices = list(range(self.a_len))
            random.shuffle(self.normal_indices)
            random.shuffle(self.abnormal_indices)
            self.length = min(len(self.normal_indices), len(self.abnormal_indices))
        else:
            self.length = len(self.file_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        try:
            if self.test_mode:
                path = self.file_list[index]
                if self.root:
                    path = os.path.join(self.root, path)
                features = np.load(path, allow_pickle=True).astype(np.float32)
                label = 0.0 if "Normal" in path else 1.0
                return torch.from_numpy(features), torch.tensor(label)

            normal_path = self.file_list[self.normal_indices[index]]
            abnormal_path = self.file_list[self.abnormal_indices[index]]

            if self.root:
                normal_path = os.path.join(self.root, normal_path)
                abnormal_path = os.path.join(self.root, abnormal_path)

            if not os.path.exists(normal_path) or not os.path.exists(abnormal_path):
                raise FileNotFoundError(f"Missing file: {normal_path} or {abnormal_path}")

            normal_features = np.load(normal_path, allow_pickle=True).astype(np.float32)
            abnormal_features = np.load(abnormal_path, allow_pickle=True).astype(np.float32)
            normal_label = 0.0 if "Normal" in normal_path else 1.0
            abnormal_label = 0.0 if "Normal" in abnormal_path else 1.0

            return (
                torch.from_numpy(normal_features),
                torch.tensor(normal_label),
                torch.from_numpy(abnormal_features),
                torch.tensor(abnormal_label)
            )
        except Exception:
            return None
