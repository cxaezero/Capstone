import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from load_data import NPYPairedDataset
from test import test
from train import train
from tqdm import tqdm
import numpy as np
import random

# ===================== 하이퍼파라미터 =====================
epochs = 10
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-5
num_workers = 0
save_path = "/weight/De_final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Triplet Loss 정의 =====================
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def distance(self, x, y):
        return torch.cdist(x, y, p=2)

    def forward(self, feats, margin=100.0):
        bs = len(feats)
        n_feats = feats[:bs // 2]
        a_feats = feats[bs // 2:]
        n_d = self.distance(n_feats, n_feats)
        a_d = self.distance(n_feats, a_feats)
        n_d_max, _ = torch.max(n_d, dim=0)
        a_d_min, _ = torch.min(a_d, dim=0)
        a_d_min = margin - a_d_min
        a_d_min = torch.max(torch.zeros(bs // 2).to(feats.device), a_d_min)
        return torch.mean(n_d_max) + torch.mean(a_d_min)

# ===================== Loss 모듈 정의 =====================
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, scores, feats, targets, alpha=0.01):
        scores = scores.squeeze()
        loss_ce = self.criterion(scores, targets)
        loss_triplet = self.triplet(feats)
        return loss_ce, alpha * loss_triplet

# ===================== collate_fn =====================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    # ===================== 모델 정의 및 데이터 로딩 =====================
    from model.classifier import Model

    classifier = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    ckpt_path = "weight/De_final_model.pth"
    classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = classifier.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_txt_path = "ucf_x3d_train_trimmed.txt"
    val_txt_path = "ucf_x3d_test_trimmed.txt"

    npy_root_train = "UCF_synth/UCF_llsynth/De_X3D_Videos"
    npy_root_val = "UCF_synth/UCF_llsynth/De_X3D_Videos_T"

    train_dataset = NPYPairedDataset(train_txt_path, root=npy_root_train, test_mode=False)
    val_dataset = NPYPairedDataset(val_txt_path, root=npy_root_val, test_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # ===================== 학습 및 검증 루프 =====================
    for epoch in range(epochs):
        train(train_loader, model, optimizer, scheduler, device, epoch)
        test(val_loader, model, device)

    # ===================== 모델 저장 =====================
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")
