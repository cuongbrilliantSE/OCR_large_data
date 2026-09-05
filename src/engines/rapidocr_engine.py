import time
import numpy as np
from typing import Dict, Any, List
from rapidocr_onnxruntime import RapidOCR
from .base_engine import BaseOCREngine
from src.utils.layout_sorter import sort_reading_order
from src.utils.text_postprocessor import TextPostProcessor


class RapidOCREngine(BaseOCREngine):
    def __init__(
        self,
        use_angle_cls: bool = True,
        det_limit_side_len: int = 960,
        min_score_thresh: float = 0.3,
        **kwargs
    ):
        self.min_score_thresh = min_score_thresh
        # Initialize RapidOCR with specified ONNX configurations
        self.ocr = RapidOCR(
            use_angle_cls=use_angle_cls,
            det_limit_side_len=det_limit_side_len,
            **kwargs
        )
        self.post_processor = TextPostProcessor(use_dictionary=True)

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs text detection & recognition on the provided image array.
        Applies column-aware reading order and Vietnamese diacritic restoration.
        """
        start_time = time.perf_counter()
        raw_result, elapse = self.ocr(image)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        if not raw_result:
            return {
                "full_text": "",
                "lines": [],
                "avg_confidence": 1.0,
                "elapse_ms": round(total_time_ms, 2)
            }

        raw_lines: List[Dict[str, Any]] = []
        confidences: List[float] = []

        for item in raw_result:
            box = item[0]
            text = str(item[1]).strip()
            score = float(item[2])

            if score >= self.min_score_thresh and text:
                raw_lines.append({
                    "text": text,
                    "confidence": round(score, 4),
                    "box": box
                })
                confidences.append(score)

        # 1. Sort lines in column-aware reading order (left column completely, then right column)
        sorted_lines = sort_reading_order(raw_lines)

        # 2. Apply Vietnamese spelling and diacritics post-processing
        cleaned_lines = []
        for line in sorted_lines:
            cleaned_text = self.post_processor.clean_text(line["text"])
            if cleaned_text:
                line["text"] = cleaned_text
                cleaned_lines.append(line)

        full_text = "\n".join([line["text"] for line in cleaned_lines])
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 1.0

        return {
            "full_text": full_text,
            "lines": cleaned_lines,
            "avg_confidence": round(avg_conf, 4),
            "elapse_ms": round(total_time_ms, 2)
        }

