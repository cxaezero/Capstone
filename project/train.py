import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from load_data import NPYPairedDataset
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np
import random
from focal_loss.focal_loss import FocalLoss

# ===================== 하이퍼파라미터 =====================
epochs = 100
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-5
num_workers = 0
save_path = "weight/De_final_model.pth"
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
        # self.criterion = nn.BCEWithLogitsLoss()
        weights = [2, 1, 1, 1, 1,
                   1, 1, 1, 1, 1]  # 클래스별 가중치
        self.criterion = FocalLoss(weights=weights, alpha=0.25, reduction='mean')
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

# ===================== 학습 함수 =====================
def train(loader, model, optimizer, scheduler, device, epoch):
    model.train()
    pred = []
    label = []

    for step, batch in tqdm(enumerate(loader), total=len(loader)):
        if batch is None:
            continue

        ninput, nlabel, ainput, alabel = batch

        input = torch.cat((ninput, ainput), dim=0).to(device)
        labels = torch.cat((nlabel, alabel), dim=0).to(device)

        scores, feats = model(input)
        pred += scores.detach().cpu().tolist()
        label += labels.detach().cpu().tolist()

        loss_func = Loss()
        loss_ce, loss_triplet = loss_func(scores, feats, labels)
        loss = loss_ce + loss_triplet

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label, pred)
    pr_auc = auc(recall, precision)
    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}, PR AUC: {pr_auc:.4f}, ROC AUC: {roc_auc:.4f}")
    return loss.item()

if __name__ == '__main__':
    # ===================== 모델 로딩 및 학습 루프 =====================
    from model.classifier import Model

    classifier = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    ckpt_path = "weight/De_final_model.pth"
    classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = classifier.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_txt_path = "ucf_x3d_train.txt"
    npy_root = "UCF_synth/UCF_llsynth/De_X3D_Videos"

    dataset = NPYPairedDataset(train_txt_path, root=npy_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    # ===================== 학습 시작 =====================
    for epoch in range(epochs):
        train(loader, model, optimizer, scheduler, device, epoch)

    # ===================== 모델 저장 =====================
    #torch.save(model.state_dict(), save_path)
    #print(f"모델 저장 완료: {save_path}")