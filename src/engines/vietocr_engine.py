import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from rapidocr_onnxruntime import RapidOCR

from .base_engine import BaseOCREngine
from src.utils.layout_sorter import sort_reading_order
from src.utils.text_postprocessor import TextPostProcessor


class VietOCREngine(BaseOCREngine):
    """
    Combines DBNet text detection (via RapidOCR) with VietOCR Transformer recognition.
    Specifically trained on Vietnamese vocabulary, characters, and diacritics.
    Runs 100% offline without requiring any API keys.
    """
    def __init__(
        self,
        model_config: str = "vgg_transformer",
        device: str = "cpu",
        min_score_thresh: float = 0.3,
        **kwargs
    ):
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        self.min_score_thresh = min_score_thresh

        # Initialize detector for bounding boxes
        self.detector = RapidOCR(use_angle_cls=True)

        # Initialize VietOCR predictor for line recognition
        config = Cfg.load_config_from_name(model_config)
        config["device"] = device
        config["predictor"]["beamsearch"] = False
        self.recognizer = Predictor(config)

        self.post_processor = TextPostProcessor(use_dictionary=True)

    def _crop_box(self, image: np.ndarray, box: list) -> Optional[Image.Image]:
        """Crop and perspective-transform bounding box to a horizontal line image."""
        try:
            pts = np.array(box, dtype="float32")
            # Order: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_w = max(int(width_a), int(width_b))

            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_h = max(int(height_a), int(height_b))

            if max_w < 5 or max_h < 5:
                return None

            dst = np.array([
                [0, 0],
                [max_w - 1, 0],
                [max_w - 1, max_h - 1],
                [0, max_h - 1]
            ], dtype="float32")

            m = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, m, (max_w, max_h))

            # Convert BGR to RGB for PIL
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(warped_rgb)
        except Exception:
            return None

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # 1. Detect text line boxes
        raw_result, _ = self.detector(image)
        if not raw_result:
            return {
                "full_text": "",
                "lines": [],
                "avg_confidence": 1.0,
                "elapse_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }

        # 2. Sort boxes by reading order
        detected_items = []
        for item in raw_result:
            box = item[0]
            det_score = float(item[2])
            if det_score >= self.min_score_thresh:
                detected_items.append({
                    "box": box,
                    "det_score": det_score,
                    "text": ""
                })

        sorted_items = sort_reading_order(detected_items)

        # 3. Recognize each text line with VietOCR
        recognized_lines = []
        confidences = []

        for item in sorted_items:
            cropped_pil = self._crop_box(image, item["box"])
            if cropped_pil is None:
                continue

            try:
                line_text = self.recognizer.predict(cropped_pil)
                line_text = self.post_processor.clean_text(line_text.strip())
                if line_text:
                    score = item["det_score"]
                    recognized_lines.append({
                        "text": line_text,
                        "confidence": round(score, 4),
                        "box": item["box"]
                    })
                    confidences.append(score)
            except Exception:
                continue

        full_text = "\n".join([line["text"] for line in recognized_lines])
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 1.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "full_text": full_text,
            "lines": recognized_lines,
            "avg_confidence": round(avg_conf, 4),
            "elapse_ms": round(elapsed_ms, 2)
        }
