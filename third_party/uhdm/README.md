# third_party/uhdm

Training / evaluation utilities copied from the ESDNet reference implementation
(https://github.com/CVMI-Lab/UHDM): `model_fn.py` (train/test step), `loss_util.py`
(VGG perceptual + L1 multi-scale loss), `metric.py` / `matlab_ssim.py` / `common.py`
(PSNR, SSIM, LPIPS, schedulers).

Nothing in `capstone/`, `scripts/` or `demo/` imports this package. It is kept only in case
ESDNet is retrained; it pulls in extra dependencies (`lpips`, `thop`, `scikit-image`) that
are not in `requirements.txt`.
