import logging
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import torchvision.utils as vutils
import cv2

# PyTorchVideo suppress logging
logging.getLogger("pytorchvideo").setLevel(logging.CRITICAL)

# 모델 로드
model_name = 'x3d_m'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
device = "cuda"
model = model.eval().to(device)
del model.blocks[-1]

from model.ESDNet import ESDNet
deweather_model = ESDNet(
    en_feature_num=48,
    en_inter_num=32,
    de_feature_num=64,
    de_inter_num=32,
    sam_number=1
).to(device)

load_path = 'weight/deweathering_model.pth'
try:
    ckpt = torch.load(load_path, map_location=device)
    state_dict = ckpt if load_path.endswith('.pth') else ckpt['state_dict']
    deweather_model.load_state_dict(state_dict)
    print("Deweathering model loaded from", load_path)
except Exception as e:
    print("Deweathering model checkpoint 로드 실패:", e)
deweather_model.eval().to(device)


# 모델 파라미터
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 15
model_transform_params = {
    "x3d_s":  {"side_size": 160, "crop_size": 160, "num_frames": 13, "sampling_rate": 6},
    "x3d_m":  {"side_size": 224, "crop_size": 224, "num_frames": 15, "sampling_rate": 5},
    "x3d_l":  {"side_size": 320, "crop_size": 320, "num_frames": 16, "sampling_rate": 5}
}
transform_params = model_transform_params[model_name]

# Permute 모듈
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)

# Transform 구성
from torchvision.transforms.v2 import CenterCrop
from torchvision.transforms import Compose, Lambda, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
)
transform_no_norm = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(transform_params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),
        ShortSideScale(size=transform_params["side_size"]),
        CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
        Permute((1, 0, 2, 3))
    ])
)
normalize = Normalize(mean, std)

# 클립 길이 계산
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

# 파일 경로 설정
test_list_path = "/content/drive/MyDrive/pipeline/Anomaly_Train.txt"
video_root = "/content/drive/MyDrive/pipeline/UCF_synth/UCF_Crimes/Videos/"
save_root = "/content/drive/MyDrive/pipeline/UCF_synth/UCF_Crimes/De_X3D_M_Videos/"
debug_root = "/content/debug_frames"

# 리스트 필터링
test_list_raw = list(open(test_list_path))
test_list = []

for path in test_list_raw:
    path = path.strip()

    video_path = os.path.join(video_root, path)
    npy_path = os.path.join(save_root, path[:-3] + 'npy')

    if os.path.isfile(npy_path):
        continue
    if not os.path.isfile(video_path):
        print(f"[경고] 영상 없음: {video_path}")
        continue

    label = 0 if 'Normal' in path else 1
    test_list.append((video_path, {'label': label, 'video_label': os.path.join(save_root, path)}))

print(f"필터링된 비디오 개수: {len(test_list)}")

# DataLoader 설정
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from torch.utils.data import DataLoader

dataset = LabeledVideoDataset(
    labeled_video_paths=test_list,
    clip_sampler=UniformClipSampler(clip_duration),
    transform=transform_no_norm,
    decode_audio=False
)
loader = DataLoader(dataset, batch_size=1)

# 저장 루프
label = None
current = None

for i, inputs in enumerate(tqdm(loader)):
    video = inputs['video'].to(device)  # [1, 3, T, H, W]
    B, C, T, H, W = video.shape
    video_frames = video.permute(0, 2, 1, 3, 4).reshape(T, C, H, W)  # [T, C, H, W]

    # Deweathering
    with torch.no_grad():
        output = deweather_model(video_frames)
        deweathered_frames = output[0] if isinstance(output, tuple) else output  # [T, C, H, W]

    # Debug: 첫 번째 프레임 원본 vs clean 저장
    if i < 10:
        os.makedirs(debug_root, exist_ok=True)
        vid_name = os.path.basename(inputs['video_label'][0]).replace('.mp4', '')
        class_name = inputs['video_label'][0].split('/')[-2]
        save_dir = os.path.join(debug_root, class_name)
        os.makedirs(save_dir, exist_ok=True)

        raw = video_frames[0].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        raw = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
        clean = deweathered_frames[0].detach().cpu().numpy().transpose(1, 2, 0)
        clean = np.clip(clean * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, f"{vid_name}_original.jpg"), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"{vid_name}_clean.jpg"), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))

    # 정규화
    video = deweathered_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]
    video = video.squeeze(0)  # [3, T, H, W]
    normalized_frames = torch.stack([
        Normalize(mean, std)(video[:, t, :, :]) for t in range(video.shape[1])
    ], dim=1)  # [3, T, H, W]
    video = normalized_frames.unsqueeze(0)  # [1, 3, T, H, W]

    # X3D feature 추출
    with torch.no_grad():
        preds = model(video).detach().cpu().numpy()

    # 저장
    for i, pred in enumerate(preds):
        new_label_path = inputs['video_label'][i][:-3]
        save_path = new_label_path + 'npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if new_label_path != label:
            if label is not None:
                np.save(label + 'npy', current.squeeze())
            label = new_label_path
            current = pred[None, ...]
        else:
            current = np.max(np.concatenate((current, pred[None, ...]), axis=0), axis=0)[None, ...]

# 마지막 저장
if label is not None:
    os.makedirs(os.path.dirname(label + 'npy'), exist_ok=True)
    np.save(label + 'npy', current.squeeze())

print("✅ 전체 완료 (디웨더링 결과는 /content/debug_frames 에 저장됨)")