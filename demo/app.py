from flask import Flask, Response, render_template
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import time
import datetime
import sys
import os.path as osp
from collections import deque
from queue import Queue
from PIL import Image 
# print(osp.dirname( osp.dirname( osp.abspath(__file__) ) ))
sys.path.append(osp.dirname( osp.dirname( osp.abspath(__file__) ) ))
import threading
from project.model.ESDNet import ESDNet
from project.model.classifier import Model

# ============= hyperparameter =============
app = Flask(__name__)
CLIP_LEN = 13
IMG_SIZE = 160
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
ANOMALY_THRESHOLD = 0.80
ALERT_FRAMES = 10
MAX_LOGS = 14
LOG_INTERVAL = 5 # seconds


# log

logs = deque(maxlen=MAX_LOGS)
logs_lock = threading.Lock()

def event_stream():
    last_index = 0
    while True:
        with logs_lock:
            if len(logs) > last_index:
                new_logs = list(logs)[:-last_index]
                last_index = len(logs)
            else:
                new_logs = []
        for log in new_logs:
            yield f"data: {log}\n\n"
        time.sleep(1)  # 1초마다 체크

# video 

transform_frame = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor()
])
normalize = Normalize(mean=MEAN, std=STD)

def load_deweather_model(device):
    deweather_model = ESDNet(
        en_feature_num=48,
        en_inter_num=32,
        de_feature_num=64,
        de_inter_num=32,
        sam_number=1
    ).to(device)

    load_path="/home/cysong/capstone/Capstone/project/weight/deweathering_model.pth"
    ckpt = torch.load(load_path, map_location=device)
    state_dict = ckpt if load_path.endswith('.pth') else ckpt['state_dict']
    deweather_model.load_state_dict(state_dict)
    deweather_model.eval()

    return deweather_model

def load_classifier_model(device):
    classifier = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)

    load_path="/home/cysong/capstone/Capstone/project/weight/De_final_model.pth"
    ckpt = torch.load(load_path, map_location=device)
    state_dict = ckpt if load_path.endswith('.pth') else ckpt['state_dict']
    classifier.load_state_dict(state_dict)
    classifier.eval()

    return classifier

def load_feature_extracter(device):
    feature_extracter = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
    feature_extracter = feature_extracter.eval().to(device)
    del feature_extracter.blocks[-1]

    return feature_extracter

def load_models(device):
    deweather = load_deweather_model(device)
    classifier = load_classifier_model(device)
    feature_ext = load_feature_extracter(device)
    return deweather, classifier, feature_ext

deweather_model_1, classifier_1, feature_extracter_1 = load_models(DEVICE)
deweather_model_2, classifier_2, feature_extracter_2 = load_models(DEVICE)


frame_queue_1 = Queue(maxsize=5)
result_queue_1 = Queue(maxsize=5)
frame_queue_2 = Queue(maxsize=5)
result_queue_2 = Queue(maxsize=5)

def inference_worker_1():
    frame_buffer = deque(maxlen=CLIP_LEN)
    anomaly_streak = 0
    last_score = 0.0

    while True:
        item = frame_queue_1.get()
        if item is None:
            break
        tensor_frame, raw_bgr = item

        with torch.no_grad():
            clean_frame, _, _ = deweather_model_1(tensor_frame)
        
        clean_np = clean_frame.squeeze(0).cpu().numpy()
        clean_np = np.transpose(clean_np, (1,2,0))
        clean_np = np.clip(clean_np * 255, 0, 255).astype(np.uint8)
        clean_np = cv2.resize(clean_np, (352, 288))
        processed_frame = cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR)

        frame_buffer.append(clean_frame.squeeze(0))
        if len(frame_buffer) == CLIP_LEN:
            clip_tensor = torch.stack(list(frame_buffer), dim=0).permute(1,0,2,3)
            normalized_clip = torch.stack([
                normalize(clip_tensor[:,t]) for t in range(CLIP_LEN)
            ], dim=1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = feature_extracter_1(normalized_clip)
                pred, _ = classifier_1(feat)
                score = torch.sigmoid(pred).item()
                last_score = score
                label = f"A: {score:.2f}"
                anomaly_streak = anomaly_streak + 1 if score >= ANOMALY_THRESHOLD else 0
        else:
            score = last_score
            label = f"A: {score:.2f}" if score else "Collection..."

        result_queue_1.put((processed_frame, label, anomaly_streak))

def inference_worker_2():
    frame_buffer = deque(maxlen=CLIP_LEN)
    anomaly_streak = 0
    last_score = 0.0

    while True:
        item = frame_queue_2.get()
        if item is None:
            break
        tensor_frame, raw_bgr = item

        with torch.no_grad():
            clean_frame, _, _ = deweather_model_2(tensor_frame)
        
        clean_np = clean_frame.squeeze(0).cpu().numpy()
        clean_np = np.transpose(clean_np, (1,2,0))
        clean_np = np.clip(clean_np * 255, 0, 255).astype(np.uint8)
        clean_np = cv2.resize(clean_np, (352, 288))
        processed_frame = cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR)

        frame_buffer.append(clean_frame.squeeze(0))
        if len(frame_buffer) == CLIP_LEN:
            clip_tensor = torch.stack(list(frame_buffer), dim=0).permute(1,0,2,3)
            normalized_clip = torch.stack([
                normalize(clip_tensor[:,t]) for t in range(CLIP_LEN)
            ], dim=1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = feature_extracter_2(normalized_clip)
                pred, _ = classifier_2(feat)
                score = torch.sigmoid(pred).item()
                last_score = score
                label = f"A: {score:.2f}"
                anomaly_streak = anomaly_streak + 1 if score >= ANOMALY_THRESHOLD else 0
        else:
            score = last_score
            label = f"A: {score:.2f}" if score else "Collection..."

        result_queue_2.put((processed_frame, label, anomaly_streak))

threading.Thread(target=inference_worker_1, daemon=True).start()
threading.Thread(target=inference_worker_2, daemon=True).start()

def generate_predefined_deweather(key):
    cap = cv2.VideoCapture("/home/cysong/capstone/Capstone/demo/demo_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 0.033 

    frame_count = 0
    processed_frame = None
    anomaly_streak = 0
    last_log_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        start_time = time.time()

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform_frame(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)

            if not frame_queue_1.full():
                frame_queue_1.put_nowait((tensor, frame.copy()))
        except Exception as e:
            print("전처리 오류:", e)

        if not result_queue_1.empty():
            processed_frame, label, anomaly_streak = result_queue_1.get()

        if processed_frame is not None:
            if anomaly_streak >= ALERT_FRAMES:
                h, w = processed_frame.shape[:2]
                cv2.rectangle(processed_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
                
                # 로그 추가
                now_time = time.time()
                if now_time - last_log_time >= LOG_INTERVAL:
                    with logs_lock:
                        now_str = datetime.datetime.now().strftime('%H:%M:%S')
                        log_msg = f"[ {now_str} - {key} ] 이상 상황 발생"
                        logs.appendleft(log_msg)
                    last_log_time = now_time  # 로그 출력 시간 갱신

            success, jpeg = cv2.imencode('.jpg', processed_frame)
            if success:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        frame_count += 1

        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)


def generate_stream_deweather(key):
    cap = cv2.VideoCapture(f"rtmp://localhost:1935/live/{key}")
    frame_count = 0
    processed_frame = None
    anomaly_streak = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform_frame(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)

            if not frame_queue_2.full():
                frame_queue_2.put_nowait((tensor, frame.copy()))
        except Exception as e:
            print("전처리 오류:", e)

        if not result_queue_2.empty():
            processed_frame, label, anomaly_streak = result_queue_2.get()

        if processed_frame is not None:
            # fps = frame_count / (time.time() - start_time)
            # cv2.putText(processed_frame, f"FPS: {fps:.2f} | {label}", (10, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if anomaly_streak >= ALERT_FRAMES:
                h, w = processed_frame.shape[:2]
                cv2.rectangle(processed_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
                
                # 로그 추가
                now_time = time.time()
                if now_time - last_log_time >= LOG_INTERVAL:
                    with logs_lock:
                        now_str = datetime.datetime.now().strftime('%H:%M:%S')
                        log_msg = f"[ {now_str} - {key} ] 이상 상황 발생"
                        logs.appendleft(log_msg)
                    last_log_time = now_time  # 로그 출력 시간 갱신

            success, jpeg = cv2.imencode('.jpg', processed_frame)
            if success:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        frame_count += 1

'''def generate_stream_original(key): # 처리 x
    cap = cv2.VideoCapture(f"rtmp://localhost:1935/live/{key}")
    frame_buffer = deque(maxlen=CLIP_LEN)
    anomlay_streak = 0
    frame_count = 0

    while True:
#        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = transform_frame(frame_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output, _, _ = deweather_model(frame_tensor)
            clean_frame = output

        clean_np = clean_frame.squeeze(0).cpu().numpy()
        clean_np = np.transpose(clean_np, (1,2,0))
        clean_np = np.clip(clean_np*255,0,255).astype(np.uint8)
        clean_np = cv2.resize(clean_np, (352, 288))
        clean_bgr = cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR)

        frame_buffer.append(clean_frame.squeeze(0))

        label = 'Collecting...'
        if len(frame_buffer) == CLIP_LEN:
            clip_tensor = torch.stack(list(frame_buffer), dim=0).permute(1,0,2,3)
            normalized_clip = torch.stack([
                normalize(clip_tensor[:,t,:,:]) for t in range(CLIP_LEN)
            ], dim=1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = feature_extracter(normalized_clip)
                pred, _ = classifier(feat)
                score = torch.sigmoid(pred).item()
                label = f"A: {score:.2f}"

                anomlay_streak = anomlay_streak + 1 if score >= ANOMALY_THRESHOLD else 0

#        elapsed = time.time() - start_time
#        fps = frame_count / elapsed if elapsed > 0 else 0
#        frame_count += 1

#        cv2.putText(clean_bgr, f"FPS: {fps:.2f} | {label}", (10, 20),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        
        if anomlay_streak >= ALERT_FRAMES:
            h, w = clean_bgr.shape[:2]
            cv2.rectangle(clean_bgr, (0,0), (w-1, h-1),(0,0,255), 3)

        _, jpeg = cv2.imencode('.jpg', clean_bgr)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')'''

def generate_stream_original(key): # 모델 처리 x
    cap = cv2.VideoCapture(f"rtmp://localhost:1935/live/{key}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/stream/<key>')
def stream_video(key):
    if key == "stream0":
        return Response(generate_predefined_deweather(key), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif key == "stream1":
        return Response(generate_stream_deweather(key), mimetype='multipart/x-mixed-replace; boundary=frame')
    else: # key == "stream2" or key == "stream3"
        return Response(generate_stream_original(key), mimetype='multipart/x-mixed-replace; boundary=frame')

# show dashboard

@app.route('/')
def dashboard():
    location = "Seoul, South Korea"
    return render_template("dashboard.html", location=location)

@app.route('/log_stream')
def log_stream():
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
