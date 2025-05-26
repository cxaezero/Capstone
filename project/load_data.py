import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc, roc_curve,precision_recall_curve
from tqdm import tqdm
import numpy as np
import random

class NPYPairedDataset(Dataset):
    def __init__(self, list_path, root=None, n_len=800, test_mode=False):
        self.test_mode = test_mode
        self.root = root
        self.list = [line.strip() for line in open(list_path) if line.strip()]
        self.n_len = n_len

        if not test_mode:
            self.a_len = len(self.list) - self.n_len
            self.n_ind = list(range(self.a_len, len(self.list)))
            self.a_ind = list(range(self.a_len))
            random.shuffle(self.n_ind)
            random.shuffle(self.a_ind)
            self.length = min(len(self.n_ind), len(self.a_ind))
        else:
            self.length = len(self.list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        try:
            if self.test_mode:
                path = self.list[index]
                if self.root is not None:
                    path = os.path.join(self.root, path)
                features = np.load(path, allow_pickle=True).astype(np.float32)
                label = 0.0 if "Normal" in path else 1.0
                return torch.from_numpy(features), torch.tensor(label)
            else:
                npath = self.list[self.n_ind[index]]
                apath = self.list[self.a_ind[index]]
                if self.root is not None:
                    npath = os.path.join(self.root, npath)
                    apath = os.path.join(self.root, apath)

                if not os.path.exists(npath) or not os.path.exists(apath):
                    raise FileNotFoundError(f"파일 없음: {npath} 또는 {apath}")

                nfeatures = np.load(npath, allow_pickle=True).astype(np.float32)
                afeatures = np.load(apath, allow_pickle=True).astype(np.float32)
                nlabel = 0.0 if "Normal" in npath else 1.0
                alabel = 0.0 if "Normal" in apath else 1.0

                return torch.from_numpy(nfeatures), torch.tensor(nlabel), torch.from_numpy(afeatures), torch.tensor(alabel)
        except Exception as e:
            return None

