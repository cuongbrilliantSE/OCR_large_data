import time
import cv2
import numpy as np
from typing import Dict, Any, List
from .base_engine import BaseOCREngine
from src.utils.layout_sorter import sort_reading_order
from src.utils.text_postprocessor import TextPostProcessor


class EasyOCREngine(BaseOCREngine):
    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = False,
        min_score_thresh: float = 0.3,
        rec_batch_size: int = 8,
        beam_width: int = 3,
        **kwargs
    ):
        import easyocr
        self.min_score_thresh = min_score_thresh
        self.rec_batch_size = rec_batch_size
        self.beam_width = beam_width
        langs = languages or ["vi", "en"]
        self.reader = easyocr.Reader(langs, gpu=gpu, **kwargs)
        self.post_processor = TextPostProcessor(use_dictionary=True)

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        # ImageCleaner returns BGR; EasyOCR expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time.perf_counter()
        raw_result = self.reader.readtext(
            image_rgb,
            batch_size=self.rec_batch_size,
            beamWidth=self.beam_width,
        )
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
            # Convert numpy box coords to native Python ints for JSON serialization
            box = [[int(pt[0]), int(pt[1])] for pt in item[0]]
            text = str(item[1]).strip()
            score = float(item[2])

            if score >= self.min_score_thresh and text:
                raw_lines.append({
                    "text": text,
                    "confidence": round(score, 4),
                    "box": box
                })
                confidences.append(score)

        # 1. Sort lines in column-aware reading order
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

