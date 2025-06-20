import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Resize, ToTensor

from model.ESDNet import ESDNet
from model.classifier import Model

CLIP_LEN = 15
IMG_SIZE = (160, 160)
BATCH_SIZE = 2
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_DIR = "/mnt/d/Capstone/data/UCF_Crimes/Videos"

mean, std = [0.45] * 3, [0.225] * 3
normalize = Normalize(mean, std)
resize = Resize(IMG_SIZE)
totensor = ToTensor()

deweather_model = ESDNet(
    en_feature_num=48,
    en_inter_num=32,
    de_feature_num=64,
    de_inter_num=32,
    sam_number=1
).to(DEVICE)
deweather_model.load_state_dict(torch.load("weight/deweathering_model.pth", map_location=DEVICE))
deweather_model.train()

x3d = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
del x3d.blocks[-1]
x3d = x3d.to(DEVICE).train()

classifier = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(DEVICE).train()

class TripletLoss(nn.Module):
    def forward(self, feats, margin=100.0):
        bs = feats.size(0)
        normal = feats[:bs // 2]
        abnormal = feats[bs // 2:]
        dist_n = torch.cdist(normal, normal)
        dist_a = torch.cdist(normal, abnormal)
        max_d_n = torch.max(dist_n, dim=0)[0]
        min_d_a = torch.min(dist_a, dim=0)[0]
        return torch.mean(max_d_n) + torch.mean(torch.clamp(margin - min_d_a, min=0))

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, scores, feats, targets, alpha=0.01):
        loss_bce = self.bce(scores.squeeze(), targets)
        loss_triplet = self.triplet(feats)
        return loss_bce + alpha * loss_triplet

class VideoDataset(Dataset):
    def __init__(self, video_dir):
        self.video_paths = glob(os.path.join(video_dir, '*/*.mp4'))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = 0 if 'Normal' in path else 1
        cap = cv2.VideoCapture(path)
        frames = []

        while len(frames) < CLIP_LEN:
            ret, frame = cap.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(IMG_SIZE)
            frames.append(totensor(image))
        cap.release()

        if len(frames) < CLIP_LEN:
            return None
        clip = torch.stack(frames[:CLIP_LEN], dim=1)
        return clip, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train(loader, deweather_model, x3d, classifier, optimizer, scheduler, loss_fn, epoch):
    deweather_model.train()
    x3d.train()
    classifier.train()

    all_preds, all_labels = [], []

    for _, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"):
        if batch is None:
            continue

        clips, targets = batch
        clips, targets = clips.to(DEVICE), targets.to(DEVICE)

        B, C, T, H, W = clips.size()
        clips = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        clean, _, _ = deweather_model(clips)
        clean = clean.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        for i in range(B):
            clean[i] = torch.stack([normalize(clean[i][:, t]) for t in range(T)], dim=1)

        feats = x3d(clean)
        scores, feat_out = classifier(feats)

        loss = loss_fn(scores, feat_out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        all_preds += torch.sigmoid(scores).detach().cpu().tolist()
        all_labels += targets.cpu().tolist()

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | PR AUC: {auc(recall, precision):.4f} | ROC AUC: {auc(fpr, tpr):.4f}")

def main():
    dataset = VideoDataset(VIDEO_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    params = list(deweather_model.parameters()) + list(x3d.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = CombinedLoss()

    for epoch in range(EPOCHS):
        train(loader, deweather_model, x3d, classifier, optimizer, scheduler, loss_fn, epoch)

    torch.save({
        'deweather': deweather_model.state_dict(),
        'x3d': x3d.state_dict(),
        'classifier': classifier.state_dict()
    }, "weight/finetuned_end2end_all.pth")

    print("Model saved.")

if __name__ == "__main__":
    main()
