"""
model_utils.py — PyTorch model loading and inference utilities for
the Vayu-Rakshak Air Quality Monitoring System.

The trained model (air_quality_model.pth) is a state_dict with architecture:
  BatchNorm1d(7) -> Linear(7, 1500) -> ReLU -> Linear(1500, 1500) x5 -> Linear(1500, 1)

Input tensor shape: (1, 7)
Feature order: [pm2p5, humidity, temp, pressure, wind, cloud, valore_originale]
Output: single float — corrected PM2.5 value (µg/m³)
"""

import os
import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_PTH = os.path.join(_BASE_DIR, "air_quality_model.pth")
_MODEL_PKL = os.path.join(_BASE_DIR, "air_quality_model", "data.pkl")

_model: Optional[nn.Module] = None


# ─────────────────────────────────────────────
# Architecture (reconstructed from state_dict)
# ─────────────────────────────────────────────

class AirQualityNet(nn.Module):
    """
    Neural network reconstructed from the saved state_dict:
      net.0  = BatchNorm1d(7)
      net.1  = Linear(7 -> 1500)
      net.2  = ReLU
      net.3  = Linear(1500 -> 1500)
      net.4  = ReLU
      ... (5 hidden layers total)
      net.13 = Linear(1500 -> 1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(7),        # 0
            nn.Linear(7, 1500),       # 1
            nn.ReLU(),                # 2
            nn.Linear(1500, 1500),    # 3
            nn.ReLU(),                # 4
            nn.Linear(1500, 1500),    # 5
            nn.ReLU(),                # 6
            nn.Linear(1500, 1500),    # 7
            nn.ReLU(),                # 8
            nn.Linear(1500, 1500),    # 9
            nn.ReLU(),                # 10
            nn.Linear(1500, 1500),    # 11
            nn.ReLU(),                # 12
            nn.Linear(1500, 1),       # 13
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

def load_model() -> nn.Module:
    """
    Load the PyTorch model. Tries .pth state_dict first, then data.pkl.
    Returns the model in eval() mode. Cached in _model singleton.
    """
    global _model
    if _model is not None:
        return _model

    # Strategy 1: load state_dict from .pth and inject into reconstructed architecture
    if os.path.isfile(_MODEL_PTH):
        try:
            logger.info(f"Loading state_dict from {_MODEL_PTH} ...")
            state_dict = torch.load(_MODEL_PTH, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict):
                net = AirQualityNet()
                net.load_state_dict(state_dict, strict=True)
                net.eval()
                _model = net
                logger.info("Model loaded from .pth state_dict + reconstructed architecture.")
                return _model
            elif isinstance(state_dict, nn.Module):
                state_dict.eval()
                _model = state_dict
                return _model
        except Exception as e:
            logger.warning(f"Strategy 1 (.pth) failed: {e}")

    # Strategy 2: try data.pkl
    if os.path.isfile(_MODEL_PKL):
        try:
            logger.info(f"Loading from {_MODEL_PKL} ...")
            loaded = torch.load(_MODEL_PKL, map_location="cpu", weights_only=False)
            if isinstance(loaded, nn.Module):
                loaded.eval()
                _model = loaded
                return _model
            elif isinstance(loaded, dict):
                net = AirQualityNet()
                net.load_state_dict(loaded, strict=True)
                net.eval()
                _model = net
                return _model
        except Exception as e:
            logger.warning(f"Strategy 2 (data.pkl) failed: {e}")

    raise FileNotFoundError(
        "Could not load the PyTorch model. "
        "Ensure 'air_quality_model.pth' exists in the project directory."
    )


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def predict(features: List[float]) -> float:
    """
    Run inference for one sample.

    Parameters
    ----------
    features : list[float] of length 7
        [pm2p5, humidity, temp, pressure, wind, cloud, valore_originale]

    Returns
    -------
    float — corrected PM2.5 prediction in µg/m³
    """
    if len(features) != 7:
        raise ValueError(f"Expected 7 features, got {len(features)}.")

    model = load_model()
    x = torch.tensor([features], dtype=torch.float32)  # (1, 7)

    with torch.no_grad():
        out = model(x)

    return float(out.squeeze().item())


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [120.0, 65.0, 27.0, 1010.0, 3.5, 40.0, 115.0]
    result = predict(sample)
    print(f"Test prediction: {result:.4f} µg/m3")
