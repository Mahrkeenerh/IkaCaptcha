"""CRNN architecture, charset, and CTC decoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Charset
# ---------------------------------------------------------------------------

CHARSET = "abcdefghjklmnpqrstuvwxy23457"  # 28 chars (0,1,6,8,9,I,O,Z excluded)
BLANK = 0
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

IMG_W = 256
IMG_H = 48

char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(2).unsqueeze(3)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


class CRNN(nn.Module):
    """4-block ResNet+SE CNN -> 1-layer BiLSTM -> CTC head. ~1.66M params."""
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            ResBlock(3, 32),
            nn.MaxPool2d(2, 2),
            ResBlock(32, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            ResBlock(64, 128),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),
            ResBlock(128, 256),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(
            input_size=256, hidden_size=hidden_size,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        out = self.dropout(rnn_out)
        out = self.fc(out)
        out = out.permute(1, 0, 2)  # (T, B, C) for CTC
        return out


# ---------------------------------------------------------------------------
# CTC decoders
# ---------------------------------------------------------------------------

def greedy_decode(logits: torch.Tensor) -> list[str]:
    """Greedy CTC decode on torch logits with shape (T, B, C)."""
    indices = logits.argmax(dim=2).permute(1, 0)  # (B, T)
    results = []
    for seq in indices:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev and idx != BLANK:
                chars.append(idx_to_char[idx])
            prev = idx
        results.append("".join(chars))
    return results


def greedy_decode_numpy(logits_tbc: np.ndarray) -> list[str]:
    """Greedy CTC decode on numpy logits with shape (T, B, C)."""
    indices = logits_tbc.argmax(axis=2).transpose(1, 0)  # (B, T)
    results = []
    for seq in indices:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev and idx != BLANK:
                chars.append(idx_to_char[idx])
            prev = idx
        results.append("".join(chars))
    return results
