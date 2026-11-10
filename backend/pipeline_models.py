"""Indicator blink and OSNet feature extraction model adapters."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import *
from .blink_model import build_model, load_weights, ModelConfig
from .pipeline_profiler import P

LOCAL_REID_REPO = Path(__file__).resolve().parent.parent / "deep-person-reid"
LOCAL_REID_PACKAGE = LOCAL_REID_REPO / "torchreid" / "__init__.py"
if not LOCAL_REID_PACKAGE.is_file():
    raise FileNotFoundError(
        "The bundled deep-person-reid package was not found. Expected "
        f"{LOCAL_REID_PACKAGE}; keep deep-person-reid inside the project folder."
    )
sys.path.insert(0, str(LOCAL_REID_REPO))
import torchreid
if Path(torchreid.__file__).resolve() != LOCAL_REID_PACKAGE.resolve():
    raise ImportError(
        "SafeRideVision must load its bundled torchreid package from "
        f"{LOCAL_REID_PACKAGE}, but loaded {torchreid.__file__}."
    )
from torchreid.utils import FeatureExtractor

device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE

cfg = ModelConfig(**BLINK_MODEL_CONFIG)
def _load_blink():
    _m = build_model(cfg)
    return load_weights(_m, BLINK_WEIGHTS, device=device)
blink_detector = P.load_model("blink", _load_blink, input="seq of 64x64 indicator crops")


# ImageNet normalization (resnet18 pretrained isi se train hua tha)
_normalize = transforms.Compose([
    transforms.ToTensor(),   # (H, W, C) numpy -> (C, H, W) tensor, scaled to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_from_track(model, track, slot_idx, device="cpu"):
    """
    Track's stored 64x64 indicator images (for ONE indicator slot — left or
    right, kept separate) -> single blink prediction for that slot.
    Returns (predicted_class, confidence) or None if that slot has no
    stored crops yet.
    """
    frames = track.getIndicatorImages(slot_idx)
    if not frames:
        return None

    with P.stage("blink.preprocess_h2d"):     # CPU normalize loop + host->GPU copy
        processed = []
        for frame in frames:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            chw = _normalize(frame)          # (C, 64, 64), normalized
            hwc = chw.permute(1, 2, 0)       # (64, 64, C) -- model ye layout expect karta hai
            processed.append(hwc)

        seq = torch.stack(processed, dim=0).unsqueeze(0).to(device)   # (1, T, 64, 64, C)
        length = torch.tensor([seq.shape[1]], device=device)
    P.note("blink.seq_len", len(frames))

    with P.stage("blink.forward"):
        model.eval()
        logits = model(seq, length)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

    return pred, conf


def extract_indicator_crop(frame, indicator_box, size=(64, 64)):
    """
    Crop indicator from frame, resize to 64x64 and return numpy array.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.
    indicator_box : list/tuple
        [confidence, x1, y1, x2, y2]
        or [x1, y1, x2, y2]
    size : tuple
        Output image size (default 64x64)

    Returns
    -------
    np.ndarray or None
        Shape: (64, 64, 3)
    """

    # Handle both formats
    if len(indicator_box) == 5:
        _, x1, y1, x2, y2 = indicator_box
    else:
        x1, y1, x2, y2 = indicator_box

    h, w = frame.shape[:2]

    # Clip coordinates
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # blink model expects RGB, frame is BGR (OpenCV)

    return crop.astype(np.uint8)


def build_osnet_encoder(model_path, model_name=OSNET_MODEL_NAME,
                         input_size=OSNET_INPUT_SIZE, device="cpu"):
    """
    Returns a callable `encoder(frame, boxes_tlwh) -> np.ndarray (N, feat_dim)`
    that matches the exact interface gdet.create_box_encoder(...) used to
    provide, so it drops straight into `features = encoder(frame, tlwhs)`
    with no other changes needed in run_video().

    Uses torchreid's FeatureExtractor, which handles resizing/normalization
    to match how the model was trained — just make sure `image_size` here
    matches what you trained OSNet at, or crops get distorted.
    """
    extractor = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        image_size=input_size,   # (H, W)
        device=device,
    )

    def encoder(frame, boxes_tlwh):
        if len(boxes_tlwh) == 0:
            return np.zeros((0, 512), dtype=np.float32)  # osnet_x1_0 default feat dim

        h_frame, w_frame = frame.shape[:2]
        crops = []
        for (x, y, w, h) in boxes_tlwh:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w_frame, int(x + w)), min(h_frame, int(y + h))
            if x2 <= x1 or y2 <= y1:
                # degenerate box (can happen if a tlwh comes in clipped to
                # nothing) — feed a black patch so the batch stays aligned
                # with boxes_tlwh instead of silently dropping an index
                crop = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        P.metric("n_reid_crops", len(crops))
        with torch.no_grad():
            with P.stage("reid.extract"):     # torchreid: CPU resize/PIL + H2D + OSNet forward
                feats = extractor(crops)                       # (N, feat_dim) torch tensor
            with P.stage("reid.postproc_d2h"):
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                out = feats.cpu().numpy().astype(np.float32)

        return out

    encoder.extractor = extractor
    return encoder


