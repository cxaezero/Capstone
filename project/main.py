import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
from load_data import NPYPairedDataset
from train import train
from test import test
from model.classifier import Model

# ===================== Configuration =====================
epochs = 10
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-5
num_workers = 0
save_path = "weight/De_final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Triplet Loss =====================
class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance(self, x, y):
        return torch.cdist(x, y, p=2)

    def forward(self, feats, margin=100.0):
        bs = len(feats)
        normal_feats = feats[:bs // 2]
        abnormal_feats = feats[bs // 2:]

        dist_n = self.distance(normal_feats, normal_feats)
        dist_a = self.distance(normal_feats, abnormal_feats)

        max_n, _ = torch.max(dist_n, dim=0)
        min_a, _ = torch.min(dist_a, dim=0)
        min_a = torch.clamp(margin - min_a, min=0)

        return torch.mean(max_n) + torch.mean(min_a)

# ===================== Combined Loss =====================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, scores, feats, targets, alpha=0.01):
        loss_bce = self.bce(scores.squeeze(), targets)
        loss_triplet = self.triplet(feats)
        return loss_bce, alpha * loss_triplet

# ===================== Collate Function =====================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ===================== Training Entry Point =====================
if __name__ == '__main__':
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_txt = "ucf_x3d_train_trimmed.txt"
    val_txt = "ucf_x3d_test_trimmed.txt"
    train_root = "UCF_synth/UCF_llsynth/De_X3D_Videos"
    val_root = "UCF_synth/UCF_llsynth/De_X3D_Videos_T"

    train_set = NPYPairedDataset(train_txt, root=train_root, test_mode=False)
    val_set = NPYPairedDataset(val_txt, root=val_root, test_mode=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    for epoch in range(epochs):
        train(train_loader, model, optimizer, scheduler, device, epoch)
        test(val_loader, model, device)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")
