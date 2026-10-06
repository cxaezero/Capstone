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
batch_size = 4
num_workers = 0
save_path = "weight/De_final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Collate Function =====================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ===================== Evaluation Function =====================
def evaluate(loader, model, device):
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader), total=len(loader), desc="Evaluating"):
            if batch is None:
                continue

            if len(batch) == 4:
                n_input, n_label, a_input, a_label = batch
                inputs = torch.cat((n_input, a_input), dim=0).to(device)
                labels = torch.cat((n_label, a_label), dim=0).to(device)
            elif len(batch) == 2:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                continue

            scores, _ = model(inputs)
            probs = torch.sigmoid(scores.squeeze())
            predictions += probs.cpu().tolist()
            targets += labels.cpu().tolist()

    predictions = np.array(predictions)
    targets = np.array(targets)
    binary_preds = (predictions >= 0.5).astype(np.float32)

    fpr, tpr, _ = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall, precision)
    accuracy = np.mean(binary_preds == targets)

    print(f"[Evaluation] Accuracy: {accuracy:.4f}, PR AUC: {pr_auc:.4f}, ROC AUC: {roc_auc:.4f}")
    return accuracy, pr_auc, roc_auc

# ===================== Execution =====================
if __name__ == '__main__':
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_txt = "ucf_x3d_test.txt"
    test_root = "UCF_synth/X3D_Videos_T"
    test_set = NPYPairedDataset(test_txt, root=test_root, test_mode=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    evaluate(test_loader, model, device)
