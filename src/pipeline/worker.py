import threading
from pathlib import Path
from typing import Dict, Any
import time

# Thread-local storage: each thread gets its own engine + cleaner instance
_local = threading.local()
_init_lock = threading.Lock()


def _get_engine_and_cleaner(config_dict: Dict[str, Any]):
    """Lazy-initialize engine and cleaner once per thread (model load is serialized)."""
    if not getattr(_local, "initialized", False):
        with _init_lock:
            if not getattr(_local, "initialized", False):
                from src.preprocessor.image_cleaner import ImageCleaner

                pre_cfg = config_dict.get("preprocessing", {})
                ocr_cfg = config_dict.get("ocr", {})
                engine_name = ocr_cfg.get("engine", "gemini").lower()

                if engine_name == "gemini":
                    from src.engines.gemini_engine import GeminiVisionEngine
                    gemini_cfg = ocr_cfg.get("gemini", {})
                    _local.engine = GeminiVisionEngine(
                        api_key=gemini_cfg.get("api_key") or None,
                        model_name=gemini_cfg.get("model_name", "gemini-2.5-flash"),
                        temperature=gemini_cfg.get("temperature", 0.0),
                        max_retries=gemini_cfg.get("max_retries", 3),
                    )
                    detector = None

                elif engine_name == "vietocr":
                    from src.engines.vietocr_engine import VietOCREngine
                    _local.engine = VietOCREngine(
                        model_config=ocr_cfg.get("vietocr_model", "vgg_transformer"),
                        device=ocr_cfg.get("vietocr_device", "cpu"),
                        min_score_thresh=ocr_cfg.get("min_score_thresh", 0.3),
                    )
                    detector = _local.engine.detector

                elif engine_name == "easyocr":
                    from src.engines.easyocr_engine import EasyOCREngine
                    _local.engine = EasyOCREngine(
                        languages=ocr_cfg.get("languages", ["vi", "en"]),
                        gpu=ocr_cfg.get("gpu", False),
                        min_score_thresh=ocr_cfg.get("min_score_thresh", 0.3),
                        rec_batch_size=ocr_cfg.get("rec_batch_size", 8),
                        beam_width=ocr_cfg.get("beam_width", 3),
                    )
                    detector = None

                else:  # rapidocr
                    from src.engines.rapidocr_engine import RapidOCREngine
                    _local.engine = RapidOCREngine(
                        use_angle_cls=ocr_cfg.get("use_angle_cls", True),
                        det_limit_side_len=ocr_cfg.get("det_limit_side_len", 960),
                        min_score_thresh=ocr_cfg.get("min_score_thresh", 0.3),
                    )
                    detector = _local.engine.ocr

                _local.cleaner = ImageCleaner(
                    max_dimension=pre_cfg.get("max_dimension", 2560),
                    auto_contrast=pre_cfg.get("auto_contrast", False),
                    auto_orient=pre_cfg.get("auto_orient", True),
                    split_double_pages=pre_cfg.get("split_double_pages", False),
                    detector=detector
                )

                _local.initialized = True

    return _local.engine, _local.cleaner


def process_image_task(task_arg: tuple) -> Dict[str, Any]:
    """
    Worker task to process one image file.
    task_arg: (input_base_dir: str, rel_path: str, config_dict: dict)
    """
    input_base_dir, rel_path, config_dict = task_arg
    abs_path = Path(input_base_dir) / rel_path
    start_time = time.perf_counter()

    try:
        engine, cleaner = _get_engine_and_cleaner(config_dict)

        img_bgr, meta = cleaner.load_and_preprocess(abs_path)

        split_double = config_dict.get("preprocessing", {}).get("split_double_pages", False)
        if split_double and meta.get("is_double_page", False):
            pages = cleaner.split_pages(img_bgr)
            parts = []
            all_lines = []
            for idx, p in enumerate(pages):
                label = "TRANG TRÁI" if idx == 0 else "TRANG PHẢI"
                res_p = engine.recognize(p)
                parts.append(f"=== [{label}] ===\n" + res_p["full_text"])
                all_lines.extend(res_p.get("lines", []))
            full_text = "\n\n".join(parts)
            confidence = 0.99
        else:
            ocr_result = engine.recognize(img_bgr)
            full_text = ocr_result["full_text"]
            all_lines = ocr_result.get("lines", [])
            confidence = ocr_result.get("avg_confidence", 1.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "file_path": rel_path,
            "success": True,
            "full_text": full_text,
            "lines": all_lines,
            "confidence": confidence,
            "processing_time_ms": round(elapsed_ms, 2),
            "error_msg": None
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "file_path": rel_path,
            "success": False,
            "full_text": "",
            "lines": [],
            "confidence": 0.0,
            "processing_time_ms": round(elapsed_ms, 2),
            "error_msg": f"{type(e).__name__}: {str(e)}"
        }
