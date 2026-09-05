from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class BaseOCREngine(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image numpy array and return recognized text and metrics.
        Returns dict with:
        {
            "full_text": str,
            "lines": List[Dict[str, Any]],  # [{"text": str, "confidence": float, "box": ...}]
            "avg_confidence": float,
            "elapse_ms": float
        }
        """
        pass
