import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
from load_data import NPYPairedDataset
from model.classifier import Model

# ===================== Configuration =====================
epochs = 100
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
        bs = feats.size(0)
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
        scores = scores.squeeze()
        loss_bce = self.bce(scores, targets)
        loss_triplet = self.triplet(feats)
        return loss_bce + alpha * loss_triplet

# ===================== Collate Function =====================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ===================== Training Function =====================
def train(loader, model, optimizer, scheduler, device, epoch):
    model.train()
    predictions, targets = [], []
    loss_fn = CombinedLoss()

    for _, batch in tqdm(enumerate(loader), total=len(loader)):
        if batch is None:
            continue

        n_input, n_label, a_input, a_label = batch
        inputs = torch.cat((n_input, a_input), dim=0).to(device)
        labels = torch.cat((n_label, a_label), dim=0).to(device)

        scores, feats = model(inputs)
        predictions += scores.detach().cpu().tolist()
        targets += labels.detach().cpu().tolist()

        loss = loss_fn(scores, feats, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    fpr, tpr, _ = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall, precision)
    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}, PR AUC: {pr_auc:.4f}, ROC AUC: {roc_auc:.4f}")
    return loss.item()

# ===================== Entry Point =====================
def main():
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_txt = "ucf_x3d_train.txt"
    train_root = "UCF_synth/De_X3D_Videos"
    train_set = NPYPairedDataset(train_txt, root=train_root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    for epoch in range(epochs):
        train(train_loader, model, optimizer, scheduler, device, epoch)

    torch.save(model.state_dict(), save_path)
    print(f"model saved: {save_path}")

if __name__ == '__main__':
    main()
