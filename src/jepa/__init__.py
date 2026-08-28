"""JEPA + looped predictor package for CIFAR-10 I-JEPA research."""

from jepa.loader import get_released_weight, list_released_weights, load_ijepa
from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor

__version__ = "0.1.0"
__all__ = [
    "IJEPA",
    "LoopedPredictor",
    "load_ijepa",
    "list_released_weights",
    "get_released_weight",
    "__version__",
]
