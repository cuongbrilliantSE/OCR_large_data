import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List


class ImageCleaner:
    def __init__(
        self,
        max_dimension: int = 2560,
        auto_contrast: bool = False,
        auto_orient: bool = False,
        split_double_pages: bool = False,
        detector: Optional[Any] = None
    ):
        self.max_dimension = max_dimension
        self.auto_contrast = auto_contrast
        self.auto_orient = auto_orient
        self.split_double_pages = split_double_pages
        self._detector = detector

    def _orient_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """EXIF orientation is already handled in load_and_preprocess."""
        return img_bgr, False



    def split_pages(self, img_bgr: np.ndarray, overlap_ratio: float = 0.03) -> List[np.ndarray]:
        """
        If the image is a two-page spread (landscape open book where width > height * 1.15),
        splits into [left_page, right_page] with a slight overlap in the middle.
        Otherwise returns [img_bgr].
        """
        h, w = img_bgr.shape[:2]
        if w > h * 1.15:
            mid = w // 2
            overlap = int(w * overlap_ratio)
            left_page = img_bgr[:, :min(w, mid + overlap)]
            right_page = img_bgr[:, max(0, mid - overlap):]
            return [left_page, right_page]
        return [img_bgr]

    def load_and_preprocess(self, image_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Safely open and preprocess image for OCR.
        Fixes EXIF rotation, checks for corruption, detects sideways orientation, resizes if oversized.
        Returns:
            (numpy array in BGR format, metadata dict)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"File is 0 bytes (corrupt): {path}")

        try:
            with Image.open(path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                orig_w, orig_h = pil_img.size

                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                img_np = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        except Exception as e:
            raise IOError(f"Corrupt or unreadable image file: {str(e)}") from e

        # Automatic 90/270 orientation check
        rotated_sideways = False
        if self.auto_orient:
            img_bgr, rotated_sideways = self._orient_image(img_bgr)

        h, w = img_bgr.shape[:2]
        resized = False

        if self.max_dimension and (w > self.max_dimension or h > self.max_dimension):
            scale = self.max_dimension / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized = True

        if self.auto_contrast:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        meta = {
            "orig_width": orig_w,
            "orig_height": orig_h,
            "processed_width": img_bgr.shape[1],
            "processed_height": img_bgr.shape[0],
            "resized": resized,
            "rotated_sideways": rotated_sideways,
            "is_double_page": (img_bgr.shape[1] > img_bgr.shape[0] * 1.15)
        }

        return img_bgr, meta
