import logging
import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.transforms.v2 import CenterCrop
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from model.ESDNet import ESDNet

logging.getLogger("pytorchvideo").setLevel(logging.CRITICAL)

model_name = 'x3d_m'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
model = model.eval().to("cuda")
del model.blocks[-1]

deweather_model = ESDNet(48, 32, 64, 32, 1).to("cuda")
load_path = 'weight/deweathering_model.pth'
try:
    ckpt = torch.load(load_path, map_location="cuda")
    state_dict = ckpt if load_path.endswith('.pth') else ckpt['state_dict']
    deweather_model.load_state_dict(state_dict)
    print("Deweathering model loaded from", load_path)
except Exception as e:
    print("Failed to load checkpoint:", e)
deweather_model.eval()

mean, std = [0.45] * 3, [0.225] * 3
fps = 15
model_transform_params = {
    "x3d_s":  {"side_size": 160, "crop_size": 160, "num_frames": 13, "sampling_rate": 6},
    "x3d_m":  {"side_size": 224, "crop_size": 224, "num_frames": 15, "sampling_rate": 5},
    "x3d_l":  {"side_size": 320, "crop_size": 320, "num_frames": 16, "sampling_rate": 5}
}
params = model_transform_params[model_name]
clip_duration = (params["num_frames"] * params["sampling_rate"]) / fps

class Permute(nn.Module):
    def __init__(self, dims): super().__init__(); self.dims = dims
    def forward(self, x): return torch.permute(x, self.dims)

transform_no_norm = ApplyTransformToKey(
    "video", Compose([
        UniformTemporalSubsample(params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),
        ShortSideScale(params["side_size"]),
        CenterCrop((params["crop_size"], params["crop_size"])),
        Permute((1, 0, 2, 3))
    ])
)

list_path = "/content/drive/MyDrive/pipeline/Anomaly_Train.txt"
video_root = "/content/drive/MyDrive/pipeline/UCF_synth/UCF_Crimes/Videos/"
save_root = "/content/drive/MyDrive/pipeline/UCF_synth/UCF_Crimes/De_X3D_M_Videos/"

test_list = []
for path in open(list_path):
    path = path.strip()
    v_path = os.path.join(video_root, path)
    n_path = os.path.join(save_root, path[:-3] + 'npy')
    if os.path.isfile(n_path): continue
    if not os.path.isfile(v_path): print("Missing:", v_path); continue
    label = 0 if 'Normal' in path else 1
    test_list.append((v_path, {'label': label, 'video_label': os.path.join(save_root, path)}))

print("Filtered videos:", len(test_list))

dataset = LabeledVideoDataset(test_list, UniformClipSampler(clip_duration), transform_no_norm, decode_audio=False)
loader = DataLoader(dataset, batch_size=1)

label, current = None, None
for i, inputs in enumerate(tqdm(loader)):
    video = inputs['video'].to("cuda")
    B, C, T, H, W = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(T, C, H, W)

    with torch.no_grad():
        output = deweather_model(frames)
        clean = output[0] if isinstance(output, tuple) else output

    video = clean.unsqueeze(0).permute(0, 2, 1, 3, 4)
    video = video.squeeze(0)
    normed = torch.stack([Normalize(mean, std)(video[:, t]) for t in range(video.size(1))], dim=1)
    video = normed.unsqueeze(0)

    with torch.no_grad():
        features = model(video).cpu().numpy()

    for i, feat in enumerate(features):
        path = inputs['video_label'][i][:-3]
        save_path = path + 'npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if path != label:
            if label is not None:
                np.save(label + 'npy', current.squeeze())
            label = path
            current = feat[None, ...]
        else:
            current = np.max(np.concatenate((current, feat[None, ...]), axis=0), axis=0)[None, ...]

if label is not None:
    os.makedirs(os.path.dirname(label + 'npy'), exist_ok=True)
    np.save(label + 'npy', current.squeeze())

print("✅ Completed.")
