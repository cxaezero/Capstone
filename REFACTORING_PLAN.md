# Capstone 리팩터링 계획서

작성일: 2026-10-06 · 결정 반영: 2026-10-06 (재학습 없음, nginx 유지) · 대상 커밋: `616186d` (main) · 범위: `project/`, `demo/`, 저장소 설정

## 0. 진행 상태 (2026-10-06)

Phase 0, 1, 2, 5를 브랜치 `refactor/phase-0-1-2-5`에서 완료했다. 아래 섹션의 파일 경로와 줄 번호는 **리팩터링 전 커밋 `616186d` 기준**이다. 현재 위치는 다음과 같다.

| 이전 | 현재 |
|---|---|
| `project/model/ESDNet.py`, `classifier.py` | `capstone/models/esdnet.py`, `classifier.py` |
| `project/load_data.py` | `capstone/data.py` |
| `project/train.py`, `test.py`, `feat_extractor.py`, `end_to_end_train.py` | `scripts/train.py`, `evaluate.py`, `extract_features.py`, `train_e2e.py` |
| `project/main.py` | 삭제 (`scripts/train.py --val-root`로 통합) |
| `project/model/model.py`, `project/utils/*.py` | `third_party/uhdm/` |
| `project/weight/`, `project/utils/weights/checkpoint_latest.tar` | `weights/` |
| `project/*.txt` | `data/lists/` |
| `demo/app.py` (단일 파일) | `demo/app.py` (라우트) + `demo/streams.py` (소스·파이프라인·브로드캐스트) |
| 손실·전처리·파이프라인 (중복) | `capstone/losses.py`, `preprocess.py`, `pipeline.py`, `evaluation.py` |

- 완료: Phase 0 전부, Phase 1 전부(A1, A3, A4, A8, A9, A10, B3, B4, B5, B7, B12, C9), Phase 2 전부(B2, B6, B8, B9, A5, A6, A7, B13, C3, C4, D1~D5 포함), Phase 5 중 결정에 따른 범위(.gitignore, .gitmodules, get-pip.py·해제된 체크포인트·미참조 mp4·`__pycache__` 제거, README). C6(로그 Condition push)와 C7(np.maximum)은 구조 정리 과정에서 함께 적용됨.
- 보류: Phase 3(C1, C2, C5), Phase 4 전부(재학습 필요), 히스토리 재작성, 가중치의 git 외부 이전.
- 검증: 리팩터링 전후 ESDNet·분류기 출력이 기존 가중치로 비트 단위 동일(`tests/`의 32개 테스트 + 수동 비교). 데모는 CPU에서 실제 모델로 첫 프레임·점수·로그 경로를 확인.

## 1. 요약

프로젝트 코드는 Python 약 1,900줄(nginx 제외)로 작지만, 아래 네 종류의 문제가 겹쳐 있어 **현재 상태로는 다른 환경에서 재현 실행이 불가능**하다.

| 분류 | 건수 | 대표 사례 |
|---|---|---|
| 🔴 확실한 버그 (실행 실패 또는 잘못된 계산) | 10 | `main.py`가 존재하지 않는 함수를 import, 분류기 `view()`가 채널/공간 축을 뒤섞음, LR 스케줄러가 배치마다 step |
| 🟠 잠재적 버그 / 설계 결함 | 15 | 학습과 데모의 전처리가 서로 다름, 데모에서 같은 스트림을 두 탭에서 열면 프레임이 섞임 |
| 🟡 비효율 | 9 | 모델 3종을 두 벌 로드, 버퍼가 찬 뒤 매 프레임 X3D 실행, VGG16을 4번 로드 |
| ⚪ 중복 / 죽은 코드 / 저장소 위생 | 7 | 손실 함수 3벌 복제, 미사용 UHDM 유틸 500줄, 대형 바이너리 약 230 MB 커밋 |

분류기 구조 버그(A2)를 고치면 기존 가중치와 호환되지 않아 재학습이 필요하다. **재학습은 하지 않기로 결정**했으므로 A2·B1(값 변경)·B10·B11은 코드를 손대지 않고 "알려진 제약"으로 문서화만 하며, 로드맵은 기존 가중치를 그대로 쓰는 범위로 한정했다. **nginx 소스·빌드 산출물도 유지**한다.

---

## 2. 발견 사항

### 🔴 A. 확실한 버그

**A1. `main.py`는 실행 자체가 불가능** — [project/main.py:12](project/main.py:12)
`from test import test`인데 [project/test.py](project/test.py)에는 `test` 함수가 없고 `evaluate`만 있다(ImportError). 모듈명 `test`는 표준 라이브러리 패키지와도 충돌한다. 또 [main.py:74-75](project/main.py:74)가 참조하는 `ucf_x3d_*_trimmed.txt`는 저장소에 없다.
→ `evaluate`로 교체, `test.py`를 `evaluate.py`로 개명, 리스트 파일 경로를 설정으로 분리.

**A2. 분류기 `DECOUPLED` 블록의 `view()`가 텐서 축을 뒤섞음** — [project/model/classifier.py:144-150](project/model/classifier.py:144)
`(B,T,H,W,C)` 텐서를 permute 없이 `.view(B*T, C, H, W)`로 재해석한다. 메모리 순서가 `[t][h][w][c]`이므로 "채널 0"에 실제 채널 0 값은 일부만 들어간다(numpy로 재현: 3×3 공간에서 9개 중 3개만 일치). 이어지는 `.view(B*H*W, C, T)`, `.view(B,T,H,W,C)`도 동일. 결과적으로 "공간 conv + 시간 conv"라는 설계 의도가 전혀 구현되지 않았고, 네트워크는 임의로 섞인 레이아웃 위에서 학습됐다.
→ 올바른 구현은 `x.permute(0,1,4,2,3).reshape(B*T,C,H,W)`이지만, `De_final_model.pth`는 뒤섞인 레이아웃으로 학습된 가중치라 수정 즉시 성능이 무너진다. **결정: 재학습하지 않으므로 수정 보류.** 코드에는 현재 동작이 의도와 다르다는 주석과 입력 레이아웃 계약(docstring)만 남기고, README '알려진 제약'에 기록한다.

**A3. 배치 크기 1에서 분류기 출력 shape 붕괴** — [project/model/classifier.py:110](project/model/classifier.py:110)
`self.pooling(x).squeeze()`가 배치 차원까지 제거해 `fc` 출력이 `(1,)`이 된다. [test.py:46-47](project/test.py:46)에서 `scores.squeeze()`가 0차원이 되어 `.tolist()`가 float을 반환하고 `predictions += float`에서 TypeError. 테스트셋 290개 ÷ 4는 나머지 2라 지금은 우연히 피해 가지만, `collate_fn`이 None 샘플을 걸러내면 언제든 배치 1이 나온다.
→ `x.flatten(1)` 또는 `squeeze(-1).squeeze(-1).squeeze(-1)`.

**A4. LR 스케줄러가 첫 epoch 안에 소진** — [project/train.py:86](project/train.py:86), [end_to_end_train.py:130](project/end_to_end_train.py:130)
`CosineAnnealingLR(T_max=epochs)`는 epoch 단위 스케줄인데 배치마다 `scheduler.step()`을 호출한다. 100 epoch 설정이면 첫 epoch의 100번째 배치에서 LR이 0에 도달하고 이후 코사인이 되돌아 오르내린다.
→ epoch 끝에서 step 하거나 `T_max = epochs * len(loader)`.

**A5. 엔드투엔드 학습의 Triplet 손실이 잘못된 가정 위에 있음** — [end_to_end_train.py:46-55](project/end_to_end_train.py:46), [76-92](project/end_to_end_train.py:76)
`TripletLoss`는 배치 앞 절반이 정상, 뒤 절반이 이상이라고 가정하지만 `VideoDataset`은 쌍 샘플링을 하지 않고 무작위 클립을 돌려준다. `BATCH_SIZE=2`이므로 "정상 1개 + 이상 1개"가 아니라 아무 조합이나 들어온다.
→ `NPYPairedDataset`처럼 (정상, 이상) 쌍을 반환하는 샘플러로 통일.

**A6. autograd 그래프 위 텐서에 in-place 대입** — [end_to_end_train.py:121](project/end_to_end_train.py:121)
`clean[i] = torch.stack(...)`는 ESDNet 출력(비-leaf)의 view에 in-place로 쓴다. backward에서 "modified by an inplace operation" 오류가 날 수 있다.
→ `(clean - mean[None,:,None,None,None]) / std[...]` 브로드캐스트 한 줄로 교체 (C3와 동일 수정).

**A7. 엔드투엔드 학습이 영상의 첫 0.5초만 사용** — [end_to_end_train.py:81-91](project/end_to_end_train.py:81)
`cap.read()`를 15번만 호출해 첫 15프레임을 클립으로 쓴다. UCF-Crime의 이상 행위는 대부분 영상 중후반에 있으므로 "이상" 라벨 클립 대부분이 실제로는 정상 장면이다. 학습이 라벨 노이즈로 무의미해진다.
→ 균일/무작위 클립 샘플링 (`feat_extractor.py`의 `UniformClipSampler` 재사용).

**A8. 절대 경로 하드코딩** — [demo/app.py:65](demo/app.py:65), [76](demo/app.py:76), [190](demo/app.py:190), [feat_extractor.py:57-59](project/feat_extractor.py:57), [end_to_end_train.py:23](project/end_to_end_train.py:23)
`/home/cysong/capstone/Capstone/...`, `/content/drive/...`(Colab), `/mnt/d/...`(WSL) 세 종류 환경의 경로가 섞여 있다. 어느 환경에서도 전부 동작하지 않는다.
→ `config.py`에서 저장소 루트 기준 상대 경로 + 환경변수/CLI 인자로 오버라이드.

**A9. 미임포트 `warnings`** — [project/utils/common.py:57](project/utils/common.py:57)
`get_lr()`이 step 밖에서 불리면 NameError. 현재 미사용 코드라 영향은 낮다(D2 참고).

**A10. 체크포인트 로드 실패를 삼키고 랜덤 가중치로 진행** — [feat_extractor.py:26-30](project/feat_extractor.py:26)
`except Exception as e: print(...)` 후 계속 실행되어 초기화되지 않은 ESDNet으로 특징을 추출·저장한다. 조용히 쓰레기 데이터가 쌓인다.
→ 예외를 그대로 raise.

### 🟠 B. 잠재적 버그 / 설계 결함

**B1. 학습·추론 전처리 불일치 (train/serve skew)** — 가장 큰 모델 품질 리스크

| 항목 | feat_extractor.py | end_to_end_train.py | demo/app.py |
|---|---|---|---|
| X3D 변형 | `x3d_m` ([16](project/feat_extractor.py:16)) | `x3d_s` ([40](project/end_to_end_train.py:40)) | `x3d_s` ([85](demo/app.py:85)) |
| 해상도 | ShortSideScale 224 + CenterCrop ([51](project/feat_extractor.py:51)) | 160×160 직접 Resize | 160×160 직접 Resize (종횡비 왜곡) |
| 클립 길이 | 15프레임, sampling_rate 5 | 15프레임 연속 | 13프레임 연속 |
| fps 가정 | 15 ([33](project/feat_extractor.py:33)), UCF-Crime은 30fps | 원본 | 원본 |

분류기가 어떤 분포로 학습됐는지 코드만으로 알 수 없고, 데모 입력은 학습 분포와 다르다.
→ **값은 바꾸지 않는다**(바꾸면 분류기 입력 분포가 달라져 재학습이 필요). 대신 `config.py`에 데모가 실제로 쓰는 값(`x3d_s`, 160×160, 13프레임, MEAN/STD)을 한 벌로 고정하고, 공용 `preprocess.py`를 통해 데모와 평가 스크립트가 같은 상수를 쓰게 한다. 특징 추출기와의 불일치는 README '알려진 제약'에 기록.

**B2. 데모의 전역 큐 구조** — [demo/app.py:101-104](demo/app.py:101)
라우트마다 전역 `frame_queue`/`result_queue` 1쌍과 워커 1개. 같은 `/stream/stream0`을 두 탭(또는 새로고침 중 겹침)에서 열면 생성기 둘이 같은 큐를 공유해 프레임이 서로 뒤섞인다.
→ "소스당 파이프라인 1개가 최신 JPEG를 보관, 클라이언트는 그것을 읽기만" 하는 생산자-브로드캐스트 구조로.

**B3. RTMP 소스 부재 시 busy loop + 핸들 누수** — [app.py:257-258](demo/app.py:257), [298-299](demo/app.py:298)
`if not ret: continue`가 sleep 없이 돌아 CPU 100%. `cap.release()`가 어디에도 없어 클라이언트가 끊겨도 핸들이 남는다.
→ 재연결 backoff, `try/finally: cap.release()`.

**B4. 워커 스레드에 예외 처리 없음** — [app.py:106-186](demo/app.py:106)
예외 한 번이면 데몬 스레드가 조용히 죽고 스트림은 마지막 프레임에서 멈춘다.
→ 루프 내부 try/except + logging, 스레드 사망 감지.

**B5. 실행 때마다 torch.hub에서 X3D 다운로드** — [app.py:85](demo/app.py:85), [end_to_end_train.py:40](project/end_to_end_train.py:40), [feat_extractor.py:17](project/feat_extractor.py:17)
`pretrained=True`로 네트워크 의존. 저장소에 있는 `project/weight/x3d_s_weights.pth`, `x3d_m_weights.pth`는 어디서도 쓰지 않는다.
→ hub는 구조만(`pretrained=False`), state_dict는 로컬에서 로드.

**B6. 데이터셋의 암묵적 순서 계약** — [project/load_data.py:8-20](project/load_data.py:8)
`n_len=800`을 하드코딩하고 리스트 파일이 "[이상 810개][정상 800개]" 순서라고 가정한다(`ucf_x3d_train.txt`는 그렇지만 `Anomaly_Train.txt`는 알파벳순이라 Normal이 중간에 있음). 라벨은 따로 경로 문자열 `"Normal"`로 판정해 진실 공급원이 둘이다. `random.shuffle`이 `__init__`에서 한 번만 실행돼 **매 epoch 같은 (정상, 이상) 쌍**이 반복되고, 이상 샘플 10개는 영원히 쓰이지 않는다.
→ 라벨로 정상/이상 인덱스를 나누고, epoch마다 재셔플(`set_epoch` 또는 `__getitem__`에서 random 선택).

**B7. 데이터 로딩 오류를 전부 삼킴** — [load_data.py:58-59](project/load_data.py:58)
`except Exception: return None` + collate에서 None 제거. 파일 누락·shape 오류가 모두 조용히 사라져 학습 데이터 수가 줄어도 알 수 없다.
→ `__init__`에서 존재 검사, `__getitem__`에서는 logging 후 raise.

**B8. 학습이 항상 기존 가중치 위에 덮어씀** — [train.py:98](project/train.py:98), [main.py:68-69](project/main.py:68)
시작 시 `save_path`를 무조건 로드(없으면 에러)하고 끝나면 같은 파일에 저장. 처음부터 학습할 수 없고, 실험 결과가 매번 덮어써진다. epoch별/best 체크포인트 없음.
→ `--resume` 플래그, `runs/<timestamp>/` 아래 저장, best 선택.

**B9. epoch 손실로 마지막 배치 값만 보고** — [train.py:93](project/train.py:93)
평균이 아니라 마지막 배치의 loss. 또 train 모드(dropout 활성)에서 AUC를 계산하므로 참고용일 뿐이다.

**B10. Triplet margin=100이 특징 스케일과 맞지 않음** — [train.py:29](project/train.py:29), [classifier.py:93,116](project/model/classifier.py:93)
분류기가 반환하는 특징은 `LayerNorm(32)` 출력이다. 감마가 커지지 않는 한 두 벡터의 L2 거리 상한은 약 11이므로 `clamp(100 - min_a, 0)`는 항상 활성이고 거의 상수다. `alpha=0.01`로 눌러 놓아 영향이 작을 뿐, 의도대로 동작하지 않는다.
→ 재학습 보류에 따라 수정하지 않음. `losses.py`로 옮길 때 주석으로 기록.

**B11. Performer 하이퍼파라미터가 의미상 부적절** — [classifier.py:6-16](project/model/classifier.py:6)
`dim=32, heads=16` → `dim_head=2`. `local_window_size=dim//8=4`는 시퀀스 길이 개념인데 채널 수에서 유도했다.
→ 재학습 보류에 따라 수정하지 않음.

**B12. 대시보드가 로그 스트림 끊김 후 영구 중단** — [demo/templates/dashboard.html:101-103](demo/templates/dashboard.html:101)
`onerror`에서 `evtSource.close()`를 호출해 EventSource의 기본 자동 재연결을 끈다. 서버 재시작 한 번이면 로그가 멈춘다. `maxLogs=24`와 서버 `MAX_LOGS=25` 불일치.

**B13. 미세조정 시 X3D BatchNorm 통계 파괴 + 저장 포맷 불일치** — [end_to_end_train.py:102](project/end_to_end_train.py:102), [152](project/end_to_end_train.py:152)
`x3d.train()`으로 배치 2짜리 통계가 사전학습 BN을 덮어쓴다. 저장 포맷도 `{'deweather','x3d','classifier'}` dict라 `demo/app.py`의 로더로는 읽을 수 없다.
→ BN freeze(`eval()` 유지 또는 `requires_grad=False`), 저장 포맷 통일.

**B14. ESDNet 입력 크기 제약(32의 배수) 미검증** — [project/model/ESDNet.py](project/model/ESDNet.py)
pixel_unshuffle(2) → stride-2 두 번 → SAM의 ¼ 스케일 때문에 H, W가 32의 배수가 아니면 CSAF에서 shape mismatch. 160, 224는 통과하지만 다른 값은 런타임 에러.
→ `assert` 또는 `utils/common.py`의 `img_pad` 재사용.

**B15. 깨진 서브모듈** — `demo/nginx-rtmp-module`
gitlink(mode 160000)로 커밋됐지만 `.gitmodules`가 없어 `git submodule update`가 동작하지 않는다. nginx 빌드를 재현할 수 없다.
→ nginx는 유지하기로 결정했으므로 디렉터리는 그대로 두고, `.gitmodules`만 복구(`url = https://github.com/arut/nginx-rtmp-module`)한다.

### 🟡 C. 비효율

**C1. 모델 3종을 두 벌 로드** — [app.py:97-98](demo/app.py:97). eval + `no_grad` 추론이므로 한 벌을 공유(필요 시 lock)하거나 워커 하나가 두 큐를 처리하면 GPU 메모리가 절반.

**C2. 버퍼가 찬 뒤 매 프레임 X3D + 분류기 실행** — [app.py:127-137](demo/app.py:127). stride 1 슬라이딩 윈도우. 분류 주기(예: 4프레임마다)를 두거나 ESDNet과 분류를 분리.

**C3. 프레임별 정규화 루프** — [app.py:129-131](demo/app.py:129), [feat_extractor.py:88](project/feat_extractor.py:88), [end_to_end_train.py:121](project/end_to_end_train.py:121). `torch.stack([normalize(x[:,t]) for t ...])` → `(x - mean) / std` 브로드캐스트 한 줄.

**C4. cv2 → PIL → Tensor 왕복** — [app.py:209-210](demo/app.py:209). `cv2.resize` + `torch.from_numpy(...).permute(2,0,1) / 255`로 PIL 제거.

**C5. 같은 프레임을 반복 JPEG 인코딩** — [app.py:218,234](demo/app.py:218). 결과 큐가 비면 이전 `processed_frame`을 다시 인코딩. 인코딩된 bytes를 캐시.

**C6. 로그 SSE 1초 폴링** — [app.py:46](demo/app.py:46). `threading.Condition`으로 push.

**C7. 클립 특징 누적을 매번 concatenate+max** — [feat_extractor.py:105](project/feat_extractor.py:105). `np.maximum(current, feat)`.

**C8. VGG16을 4번 로드, mean/std가 학습 파라미터** — [utils/loss_util.py:33-43](project/utils/loss_util.py:33). 한 번 로드 후 슬라이스, `register_buffer`. (현재 미사용 코드)

**C9. `torch.load`에 `weights_only` 미지정** — 전역. torch 2.6부터 기본값이 바뀌어 경고/오류 가능. state_dict만 저장하므로 `weights_only=True` 명시.

### ⚪ D. 중복 / 죽은 코드 / 저장소 위생

**D1. 중복 정의**
- `TripletLoss`, `CombinedLoss`, `collate_fn`이 [train.py](project/train.py), [main.py](project/main.py), [end_to_end_train.py](project/end_to_end_train.py)에 각각 존재. `main.py`의 `CombinedLoss`는 반환 타입까지 다르고(튜플) 실제로는 쓰이지 않는다.
- [app.py](demo/app.py)의 `inference_worker_1`/`_2`(106-186)는 완전히 동일, `generate_predefined_deweather`/`generate_stream_deweather`(189-292)는 95% 동일.
- [ESDNet.py](project/model/ESDNet.py)의 `DB`와 `RDB`는 residual 유무만 다르다.

**D2. 죽은 코드**
- [project/model/model.py](project/model/model.py), [project/utils/common.py](project/utils/common.py), [loss_util.py](project/utils/loss_util.py), [metric.py](project/utils/metric.py), [matlab_ssim.py](project/utils/matlab_ssim.py): UHDM(ESDNet 원본) 잔재. 어느 학습·데모 스크립트도 import하지 않으며, `lpips`, `thop`, `skimage` 같은 무거운 의존성만 끌고 온다.
- `ESDNet._initialize_weights` 미호출. `app.py`의 `raw_bgr`, `frame_count`, `label` 미사용. `main.py`의 `CombinedLoss`.
- `project/utils/weights/checkpoint_latest.tar`(68 MB)와 그 압축 해제본 1,276파일(또 68 MB): 코드 어디서도 로드하지 않음. `888tiny.pkl`도 미사용.

**D3. import 루트 불일치**
`project/` 스크립트는 `from model.classifier import ...`(cwd가 `project/`여야 함), `demo/app.py`는 `sys.path.append` 후 `from project.model...`. `utils/`에는 `__init__.py`가 없다.
→ `capstone/` 패키지 + `pyproject.toml`(editable install)로 통일.

**D4. 설정과 로깅**
모든 하이퍼파라미터·경로가 모듈 상단 상수 또는 리터럴. `print` 로깅. CLI 인자 없음.
→ `config.py`(dataclass) + argparse, `logging`.

**D5. 네이밍**
`feature_extracter`(오타), `test.py`(stdlib 충돌), `label`이 경로를 담는 변수([feat_extractor.py:94-105](project/feat_extractor.py:94)), `weight/`와 `utils/weights/` 두 디렉터리, `De_final_model.pth`/`888tiny.pkl`(같은 크기, 용도 불명), `Model`(너무 일반적).

**D6. 저장소 위생**
- `.gitignore` 없음 → `.DS_Store`, `__pycache__`, 비어 있지 않은 디렉터리의 `.gitkeep`까지 커밋.
- 커밋된 대형 바이너리: 체크포인트 tar + 해제본(136 MB), 추론 가중치 4개(약 47 MB), mp4 2개(20 MB), `get-pip.py`(2.2 MB), nginx 소스 + `objs/*.o` + 6.7 MB 바이너리(661파일). **nginx는 유지 결정**, 나머지가 정리 대상.
- `requirements.txt`: Colab 전체 `pip freeze` 631줄. `+cu124` 핀 때문에 macOS/CPU에서 설치 불가.

**D7. 테스트 0개, CI 없음, README에 실행 방법 없음.**

---

## 3. 리팩터링 로드맵

재학습을 하지 않으므로 모든 단계가 **기존 가중치를 그대로 쓰면서** 적용된다. 재학습이 필요한 항목은 Phase 4에 '보류'로 모아 두었고 코드를 손대지 않는다.

### Phase 0 — 안전망 (동작 변화 없음)
1. `.gitignore` 추가 (`__pycache__/`, `.DS_Store`, `runs/`, `*.npy`). `demo/nginx-1.25.2/`는 유지하므로 ignore하지 않는다. 이미 추적 중인 가중치·mp4를 어떻게 할지는 Phase 5에서 결정.
2. `requirements.txt`를 실제 필요한 패키지로 축소: `torch`, `torchvision`, `pytorchvideo`, `performer-pytorch`, `flask`, `opencv-python`, `scikit-learn`, `pillow`, `numpy`, `tqdm`. CUDA 핀은 `requirements-cuda.txt`로 분리.
3. `capstone/config.py` 신설: 저장소 루트 기준 경로, 모델 하이퍼파라미터(`X3D_VARIANT`, `CLIP_LEN`, `IMG_SIZE`, `MEAN`, `STD`, `ANOMALY_THRESHOLD` …)를 한 곳에. 환경변수로 오버라이드.
4. `pyproject.toml`로 패키지화, `pip install -e .`.
5. 최소 테스트: 랜덤 텐서로 `ESDNet`/`Model` forward shape 검사(배치 1 포함), `NPYPairedDataset` 로딩, Flask 라우트 200 응답.

### Phase 1 — 즉시 수정 버그 (호환 유지)
A1, A3, A4, A8, A9, A10, B3, B4, B5, B7, B12, C9. 전부 가중치와 무관하며 각각 수 줄 수정.

### Phase 2 — 중복 제거와 구조 정리
1. `capstone/losses.py`: `TripletLoss`, `CombinedLoss` 1벌. `capstone/data.py`: `NPYPairedDataset` + `collate_fn` + 쌍 샘플링 `VideoDataset`(B6 수정 포함).
2. `capstone/preprocess.py`: 프레임 → 텐서, 클립 정규화(C3, C4)를 학습·추출·데모가 공용.
3. `capstone/pipeline.py`: "프레임 → ESDNet → 클립 버퍼 → X3D → 분류기 → 점수"를 클래스 하나로. `demo/app.py`의 워커 2개와 생성기 2개가 이 클래스의 인스턴스가 된다(B2 구조 변경 포함).
4. 죽은 UHDM 코드(D2)는 `third_party/uhdm/`로 이동하거나 삭제. ESDNet 재학습 계획이 없으면 삭제 권장.
5. `test.py` → `evaluate.py`, 스크립트는 `scripts/` 아래로, 각각 argparse 진입점.
6. `train.py`에 `--resume`, `runs/<timestamp>/` 체크포인트, best 선택, 평균 손실(B8, B9).
7. `end_to_end_train.py`의 스크립트 버그 수정(A5 쌍 샘플링, A6 브로드캐스트 정규화, A7 균일 클립 샘플링, B13 BN freeze·저장 포맷). 학습을 실행하지는 않으므로 검증은 더미 클립으로 forward/backward 1 step만 수행.

### Phase 3 — 데모 성능
C1(모델 공유), C2(분류 stride), C5(JPEG 캐시), C6(Condition push). Apple Silicon에서 돌리려면 `DEVICE` 선택에 `mps` 추가.

### Phase 4 — 보류: 재학습이 필요한 항목 (결정: 진행하지 않음)
아래는 고치면 `De_final_model.pth`와 호환이 깨지는 항목이다. 코드를 수정하지 않고 README '알려진 제약' 섹션과 해당 코드의 주석에만 기록한다. 나중에 학습 데이터와 GPU 여건이 생기면 이 목록부터 착수한다.
1. A2 `DECOUPLED` 축 뒤섞임 — 올바른 permute 구현을 주석으로 병기.
2. B1 특징 추출기(`x3d_m`, 224)와 데모(`x3d_s`, 160)의 전처리 불일치 — 값 통일 보류.
3. B10 Triplet margin=100, B11 Performer `dim_head=2` 설정.

### Phase 5 — 저장소 위생
1. 대형 바이너리 제거. 가중치는 GitHub Release 에셋 + `scripts/download_weights.sh`, 또는 Git LFS.
2. `demo/nginx-1.25.2/`는 **유지**(결정). `.gitmodules`만 복구해 `nginx-rtmp-module` 서브모듈이 받아지게 하고, README에 configure/make 순서를 적는다(B15).
3. 중복 체크포인트 해제본, `get-pip.py`, 샘플 mp4 제거.
4. 히스토리 재작성(`git filter-repo`) 여부 결정. 공개 저장소이므로 force push 영향을 팀과 합의.
5. README에 설치·가중치 다운로드·특징 추출·학습·데모 실행 순서 추가.

### 목표 구조

```
Capstone/
├── pyproject.toml
├── requirements.txt            # CPU 기본
├── requirements-cuda.txt
├── .gitignore
├── capstone/
│   ├── config.py               # 경로·하이퍼파라미터 단일 소스
│   ├── models/
│   │   ├── esdnet.py
│   │   ├── classifier.py       # DECOUPLED 수정 + legacy 플래그
│   │   └── x3d.py              # hub 구조 + 로컬 가중치
│   ├── data.py                 # NPYPairedDataset, VideoDataset, collate
│   ├── losses.py
│   ├── preprocess.py
│   └── pipeline.py             # 프레임→점수 (학습·데모 공용)
├── scripts/
│   ├── extract_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_e2e.py
│   └── download_weights.sh
├── demo/
│   ├── app.py                  # 라우트만
│   ├── streams.py              # 소스별 파이프라인 + 브로드캐스트
│   └── templates/dashboard.html
├── data/lists/*.txt
├── weights/                    # gitignored
├── tests/
└── third_party/uhdm/           # 또는 삭제
```

---

## 4. 검증 방법

- **단위**: `pytest tests/` — 모델 forward shape(배치 1, 4), 데이터셋 길이·라벨, 손실 값이 유한.
- **A4 검증**: 첫 epoch 동안 `optimizer.param_groups[0]['lr']`를 로그로 찍어 단조 감소하지 않는 구간이 없는지 확인.
- **회귀 확인**: `scripts/evaluate.py`로 기존 `De_final_model.pth`의 테스트 AUC를 먼저 기록(기준선). 가중치를 바꾸지 않으므로 Phase 1~3 후에도 **같은 값**이 나와야 하며, 달라지면 전처리나 모델 코드가 의도치 않게 바뀐 것이다.
- **데모**: `demo_video.mp4`로 `/stream/stream0` 1분 재생, 같은 URL 두 탭 동시 열기, RTMP 없이 `/stream/stream1` 열어 CPU 사용률 확인, 서버 재시작 후 로그 패널 복구 확인.

## 5. 결정 사항

**확정 (2026-10-06)**
1. **재학습 없음.** A2, B1 값 변경, B10, B11은 수정하지 않고 알려진 제약으로 문서화(Phase 4 보류).
2. **nginx 유지.** `demo/nginx-1.25.2/`를 삭제하거나 Docker로 대체하지 않는다. `.gitmodules`만 복구.

**미결 (기본값으로 진행하되 이견 있으면 알려 주세요)**
3. **히스토리 재작성 여부.** 기본값: 하지 않음. 앞으로의 커밋에서만 대형 파일을 제거한다. 과거 커밋까지 지우려면 force push가 필요해 팀 합의가 먼저다.
4. **UHDM 유틸(D2) 삭제 vs 보관.** 기본값: `third_party/uhdm/`로 이동. 삭제보다 되돌리기 쉽고, ESDNet 재학습 여지를 남긴다.
5. **추적 중인 가중치(.pth) 처리.** 기본값: 당분간 git에 그대로 둔다. Release 에셋이나 Git LFS로 옮기려면 업로드 권한이 필요하다.
