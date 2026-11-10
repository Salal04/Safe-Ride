"""
blinker.py
----------
CNN + Sequence-model hybrid classifier module.

A CNN backbone (from torchvision) extracts per-frame features, which are then
fed into a sequence model (LSTM / GRU / BiLSTM / RNN / Transformer) to
produce a fixed-size embedding, followed by a final classification head.

Usage
-----
    from blinker import build_model, ModelConfig

    cfg = ModelConfig(cnn_name="resnet18", seq_type="LSTM",
                       num_layers=2, hidden_size=256, dropout=0.5)
    model = build_model(cfg)
    logits = model(video_batch, lengths)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

__all__ = [
    "ModelConfig",
    "SequenceClassifier",
    "LSTMClassifier",
    "GRUClassifier",
    "BiLSTMClassifier",
    "TransformerClassifier",
    "RNNClassifier",
    "CombinedModel",
    "build_model",
    "load_weights",
]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """All hyperparameters needed to build a CombinedModel in one place."""

    cnn_name: str = "resnet18"          # any classification model name in torchvision.models
    seq_type: str = "LSTM"              # one of SEQUENCE_REGISTRY keys
    num_layers: int = 2
    hidden_size: int = 256
    dropout: float = 0.5
    seq_embed_dim: int = 128            # output dim of the sequence classifier's fc layer
    num_classes: int = 2                # final classification head size
    freeze_cnn: bool = False            # optionally freeze the CNN backbone


# --------------------------------------------------------------------------- #
# Sequence classifiers
# --------------------------------------------------------------------------- #
class SequenceClassifier(nn.Module, ABC):
    """Common interface every sequence model in this module must implement."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int,
                 dp_out: float, embed_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """x: (B, T, input_dim), lengths: (B,) -> returns (B, embed_dim)."""
        raise NotImplementedError


class LSTMClassifier(SequenceClassifier):
    def __init__(self, input_dim, hidden_size, num_layers, dp_out, embed_dim=128):
        super().__init__(input_dim, hidden_size, num_layers, dp_out, embed_dim)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                             batch_first=True, dropout=dp_out)
        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


class GRUClassifier(SequenceClassifier):
    def __init__(self, input_dim, hidden_size, num_layers, dp_out, embed_dim=128):
        super().__init__(input_dim, hidden_size, num_layers, dp_out, embed_dim)
        self.gru = nn.GRU(input_dim, hidden_size, num_layers,
                           batch_first=True, dropout=dp_out)
        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.fc(h_n[-1])


class BiLSTMClassifier(SequenceClassifier):
    def __init__(self, input_dim, hidden_size, num_layers, dp_out, embed_dim=128):
        super().__init__(input_dim, hidden_size, num_layers, dp_out, embed_dim)
        self.bilstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True,
                               bidirectional=True, dropout=dp_out)
        self.fc = nn.Linear(hidden_size * 2, embed_dim)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.bilstm(packed)
        # last layer: forward direction = h_n[-2], backward direction = h_n[-1]
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(x)


class RNNClassifier(SequenceClassifier):
    def __init__(self, input_dim, hidden_size, num_layers, dp_out, embed_dim=128):
        super().__init__(input_dim, hidden_size, num_layers, dp_out, embed_dim)
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers,
                           batch_first=True, dropout=dp_out)
        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        return self.fc(h_n[-1])


class TransformerClassifier(SequenceClassifier):
    def __init__(self, input_dim, hidden_size, num_layers, dp_out, embed_dim=128, nhead=8):
        super().__init__(input_dim, hidden_size, num_layers, dp_out, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead,
                                            batch_first=True, dropout=dp_out)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        B, T, _ = x.shape
        # True at padded positions so the transformer ignores them
        mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.to(x.device).unsqueeze(1)
        x = self.transformer(x, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        x = (x * valid).sum(dim=1) / valid.sum(dim=1)
        return self.fc(x)


# Registry: add a new sequence model here and it's instantly usable via ModelConfig.seq_type
SEQUENCE_REGISTRY: Dict[str, Type[SequenceClassifier]] = {
    "LSTM": LSTMClassifier,
    "GRU": GRUClassifier,
    "BiLstm": BiLSTMClassifier,
    "RNN": RNNClassifier,
    "Transformer": TransformerClassifier,
}


# --------------------------------------------------------------------------- #
# Combined model
# --------------------------------------------------------------------------- #
class CombinedModel(nn.Module):
    """CNN backbone (per-frame features) -> sequence classifier -> FC head."""

    def __init__(self, cnn_model: nn.Module, sequence_classifier: SequenceClassifier,
                 cnn_output_dim: int, num_classes: int = 2):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.cnn = cnn_model
        self.seq = sequence_classifier
        self.fc = nn.Linear(sequence_classifier.embed_dim, num_classes)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, self.cnn_output_dim)
        x = self.seq(x, lengths)
        return self.fc(x)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def _build_cnn_backbone(cnn_name: str, freeze: bool = False) -> tuple[nn.Module, int]:
    """Loads a torchvision classification model, strips its head, returns
    (backbone, output_feature_dim)."""
    if not hasattr(models, cnn_name):
        raise ValueError(
            f"Unknown cnn_name '{cnn_name}'. Must be a model available in torchvision.models."
        )
    model_fn = getattr(models, cnn_name)
    cnn_model = model_fn(weights="DEFAULT")
    cnn_model.fc = nn.Identity()  # strip ImageNet classifier head, use raw feature vector

    with torch.no_grad():
        cnn_output_dim = cnn_model(torch.zeros(1, 3, 64, 64)).shape[1]

    if freeze:
        for p in cnn_model.parameters():
            p.requires_grad_(False)

    return cnn_model, cnn_output_dim


def build_model(cfg: ModelConfig) -> CombinedModel:
    """Builds a CombinedModel from a ModelConfig.

    Raises:
        ValueError: if cfg.seq_type isn't a registered sequence model, or
                    cfg.cnn_name isn't a valid torchvision model.
    """
    if cfg.seq_type not in SEQUENCE_REGISTRY:
        valid = ", ".join(SEQUENCE_REGISTRY)
        raise ValueError(f"Unknown seq_type '{cfg.seq_type}'. Valid options: {valid}")

    cnn_model, cnn_output_dim = _build_cnn_backbone(cfg.cnn_name, cfg.freeze_cnn)

    seq_cls = SEQUENCE_REGISTRY[cfg.seq_type]
    seq_model = seq_cls(
        input_dim=cnn_output_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dp_out=cfg.dropout,
        embed_dim=cfg.seq_embed_dim,
    )

    return CombinedModel(cnn_model, seq_model, cnn_output_dim, num_classes=cfg.num_classes)


# ----------------------------------------------------------------------- #
# Old checkpoints (saved back when LSTMClassifier's LSTM layer was named
# `self.rnn` instead of today's `self.lstm`) store their keys as
# "seq.rnn.*" -- the architecture is identical (still an nn.LSTM), only
# the attribute name changed in a later refactor. Same idea applies to
# any other sequence classifier that gets renamed down the line: add the
# old_name -> new_name pair here and old checkpoints keep loading with no
# other code changes needed.
# ----------------------------------------------------------------------- #
_LEGACY_KEY_RENAMES = {
    "seq.rnn.": "seq.lstm.",
}


def _remap_legacy_keys(state_dict: dict) -> dict:
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in _LEGACY_KEY_RENAMES.items():
            if new_k.startswith(old_prefix):
                new_k = new_prefix + new_k[len(old_prefix):]
                break
        remapped[new_k] = v
    return remapped


def load_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Loads saved weights into a model (in place) and returns it.

    Handles both plain state_dicts and checkpoints that wrap the state_dict
    under a "model_state_dict" / "state_dict" key (common when the checkpoint
    also stores optimizer state, epoch number, etc.).

    Also handles OLDER checkpoints saved under a previous attribute naming
    scheme (e.g. "seq.rnn.*" from before the LSTM layer was renamed to
    "seq.lstm.*") -- see _LEGACY_KEY_RENAMES / _remap_legacy_keys() above.
    The first load attempt uses the checkpoint's keys as-is; only if that
    raises a key-mismatch error do we retry once with the legacy names
    remapped. If the remapped load ALSO fails, the original error is
    re-raised so a genuinely incompatible checkpoint doesn't fail silently.

    Args:
        model: an already-built model (e.g. from build_model(cfg)).
        checkpoint_path: path to a .pt / .pth file.
        device: where to map the loaded tensors ("cpu", "cuda", "cuda:0", ...).
        strict: if False, allows partial loading (e.g. CNN weights only,
                or resuming after adding/removing a layer). Mismatched keys
                are reported instead of raising.

    Returns:
        The same model, with weights loaded, moved to `device` and in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # assume it's already a raw state_dict

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        remapped_state_dict = _remap_legacy_keys(state_dict)
        if remapped_state_dict == state_dict:
            # Nothing to remap -- this isn't the legacy-naming case, so the
            # original error is the real problem. Don't hide it.
            raise
        print(f"[load_weights] strict load failed ({e}); retrying with legacy "
              f"key names remapped ({list(_LEGACY_KEY_RENAMES.items())})...")
        missing, unexpected = model.load_state_dict(remapped_state_dict, strict=strict)
        print("[load_weights] legacy-remapped load succeeded.")

    if not strict and (missing or unexpected):
        print(f"[load_weights] missing keys: {missing}")
        print(f"[load_weights] unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


# Backward-compatible alias for the old function name/signature
def model_Genrator(cnn_name, seq, num_lay, hd_size=256, drop_out=0.5) -> CombinedModel:
    """Deprecated: use build_model(ModelConfig(...)) instead."""
    cfg = ModelConfig(cnn_name=cnn_name, seq_type=seq, num_layers=num_lay,
                       hidden_size=hd_size, dropout=drop_out)
    return build_model(cfg)


# --------------------------------------------------------------------------- #
# Script entry point (only runs when this file is executed directly,
# not when it's imported as a module)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = ModelConfig(
        cnn_name="resnet18",
        seq_type="LSTM",
        num_layers=2,
        hidden_size=256,
        dropout=0.5,
    )
    model = build_model(cfg)
    print(model)
