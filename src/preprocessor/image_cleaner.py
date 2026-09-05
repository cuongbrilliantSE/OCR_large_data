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
        auto_orient: bool = True,
        split_double_pages: bool = False,
        detector: Optional[Any] = None
    ):
        self.max_dimension = max_dimension
        self.auto_contrast = auto_contrast
        self.auto_orient = auto_orient
        self.split_double_pages = split_double_pages
        self._detector = detector

    def _get_detector(self):
        if self._detector is None:
            from rapidocr_onnxruntime import RapidOCR
            self._detector = RapidOCR(use_angle_cls=True)
        return self._detector

    def _orient_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detect if text lines in image are oriented vertically (taken sideways at 90/270 degrees).
        Rotates image so text lines are horizontal for optimal OCR detection.
        """
        try:
            h, w = img_bgr.shape[:2]
            scale = 960 / max(h, w)
            small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

            detector = self._get_detector()
            boxes, _ = detector(small)
            if not boxes:
                return img_bgr, False

            horiz_count = 0
            vert_count = 0
            for box, text, score in boxes:
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)
                if bw > bh * 1.5:
                    horiz_count += 1
                elif bh > bw * 1.5:
                    vert_count += 1

            # If text lines are predominantly vertical, the image was captured sideways
            if vert_count > horiz_count and vert_count >= 3:
                # Test 90 CW vs 90 CCW
                img_90 = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                small_90 = cv2.resize(img_90, (int(h * scale), int(w * scale)))
                b90, _ = detector(small_90)

                img_270 = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                small_270 = cv2.resize(img_270, (int(h * scale), int(w * scale)))
                b270, _ = detector(small_270)

                # Count valid horizontal lines detected in each orientation
                def count_valid_horiz(b_list):
                    if not b_list:
                        return 0, 0.0
                    cnt = 0
                    total_score = 0.0
                    for box, text, score in b_list:
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        if (max(xs) - min(xs)) > (max(ys) - min(ys)) * 1.2:
                            cnt += 1
                            total_score += score
                    return cnt, total_score

                cnt_90, score_90 = count_valid_horiz(b90)
                cnt_270, score_270 = count_valid_horiz(b270)

                if cnt_90 > cnt_270 or (cnt_90 == cnt_270 and score_90 >= score_270):
                    return img_90, True
                else:
                    return img_270, True

            return img_bgr, False
        except Exception:
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
