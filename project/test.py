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

# ===================== 하이퍼파라미터 =====================
batch_size = 4
num_workers = 0
save_path = "weight/De_final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model.classifier import Model

classifier = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
ckpt_path = "weight/De_final_model.pth"
classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
model = classifier.to(device)

# ===================== collate_fn =====================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def test(loader, model, device):
    model.eval()
    pred = []
    label = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader), total=len(loader), desc="Testing"):
            if batch is None:
                continue

            if len(batch) == 4:  # 학습용 구조 (n, a 쌍)
                ninput, nlabel, ainput, alabel = batch
                input = torch.cat((ninput, ainput), dim=0).to(device)
                labels = torch.cat((nlabel, alabel), dim=0).to(device)
            elif len(batch) == 2:  # 테스트 구조 (단일 샘플)
                input, labels = batch
                input = input.to(device)
                labels = labels.to(device)
            else:
                print(f"[경고] 예상치 못한 배치 형태: len(batch) = {len(batch)}")
                continue

            scores, _ = model(input)
            probs = torch.sigmoid(scores.squeeze())
            pred += probs.cpu().tolist()
            label += labels.cpu().tolist()

    pred = np.array(pred)
    label = np.array(label)
    pred_label = (pred >= 0.5).astype(np.float32)

    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label, pred)
    pr_auc = auc(recall, precision)
    acc = np.mean(pred_label == label)

    print(f"[Test Result] Accuracy: {acc:.4f}, PR AUC: {pr_auc:.4f}, ROC AUC: {roc_auc:.4f}")
    return acc, pr_auc, roc_auc

if __name__ == '__main__':
    # ===================== 테스트셋 경로 설정 및 로딩 =====================
    test_txt_path = "ucf_x3d_test_trimmed.txt"
    npy_root = "UCF_synth/UCF_llsynth/X3D_Videos_T"

    test_dataset = NPYPairedDataset(test_txt_path, root=npy_root, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # ===================== 모델 불러오기 및 테스트 실행 =====================
    model.load_state_dict(torch.load(save_path, map_location=device))  # 학습 후 저장된 모델
    model = model.to(device)
    test(test_loader, model, device)