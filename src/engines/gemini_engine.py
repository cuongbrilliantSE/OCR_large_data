import os
import io
import re
import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .base_engine import BaseOCREngine

# Load environment variables from .env if present
load_dotenv()

SYSTEM_PROMPT = """Bạn là một chuyên gia số hóa tài liệu và sách cổ, tài liệu lịch sử tiếng Việt với độ chính xác tuyệt đối 100%.
Nhiệm vụ: Trích xuất toàn bộ văn bản trong ảnh một cách trung thực và chính xác nhất.

Quy tắc bắt buộc:
1. Đảm bảo đúng 100% tiếng Việt có dấu, đầy đủ dấu thanh, dấu mũ, dấu móc, không nuốt khoảng trắng giữa các từ.
2. Giữ nguyên toàn bộ cấu trúc văn bản:
   - Các tiêu đề chính in hoa (ví dụ: BAN CHỈ ĐẠO BIÊN SOẠN, LỜI GIỚI THIỆU).
   - Thứ tự các mục số La Mã (I, II, III), danh sách đánh số (1., 2., 3.), gạch đầu dòng.
   - Họ tên, chức vụ, đơn vị công tác giữ đúng chữ in hoa hoặc in thường như trong sách.
   - Các đoạn văn giữ nguyên ngắt dòng, ngắt đoạn logic.
3. Nếu ảnh chụp mở cả 2 trang sách (trang trái và trang phải), hãy phân định rõ ràng như sau:
=== [TRANG TRÁI] ===
(Nội dung toàn bộ trang bên trái, bao gồm cả số trang nếu có)

=== [TRANG PHẢI] ===
(Nội dung toàn bộ trang bên phải, bao gồm cả số trang nếu có)
4. Giữ nguyên các số trang được in ở cuối hoặc đầu trang (ví dụ: 4, 5).
5. Chỉ xuất trực tiếp nội dung văn bản đã trích xuất, KHÔNG thêm lời mở đầu, KHÔNG thêm chú thích, KHÔNG bọc toàn bộ bằng khối ```markdown.
"""


class GeminiVisionEngine(BaseOCREngine):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        max_retries: int = 8,
        **kwargs
    ):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.model_pool = list(dict.fromkeys([
            model_name,
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-flash-latest"
        ]))
        self.current_model_idx = 0
        self.model_name = self.model_pool[0]
        self.temperature = temperature
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "Chưa cấu hình GEMINI_API_KEY. Vui lòng thiết lập biến môi trường GEMINI_API_KEY "
                "hoặc điền vào configs/config.yaml hoặc file .env."
            )

        self._init_client()

    def _init_client(self):
        """Initialize either google-genai or google.generativeai client."""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._sdk_type = "genai"
        except Exception:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._genai_legacy = genai_legacy
            self._legacy_models: Dict[str, Any] = {}
            self.client = self._get_legacy_model(self.model_name)
            self._sdk_type = "legacy"

    def _get_legacy_model(self, model_name: str):
        """Return (and cache) a legacy GenerativeModel for the given model name.

        The model id is bound at construction time in the legacy SDK, so the
        pool-switching logic in _call_gemini must rebuild the model when it
        changes rather than reuse the one made at init.
        """
        if model_name not in self._legacy_models:
            self._legacy_models[model_name] = self._genai_legacy.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
            )
        return self._legacy_models[model_name]

    def _call_gemini(self, pil_img: Image.Image) -> str:
        """Call Gemini Vision API with automatic retry on rate limits."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if self._sdk_type == "genai":
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[pil_img, SYSTEM_PROMPT],
                    )
                    return (response.text or "").strip()
                else:
                    model = self._get_legacy_model(self.model_name)
                    response = model.generate_content(
                        [pil_img, "Hãy đọc toàn bộ văn bản trong ảnh theo đúng quy tắc đã hướng dẫn."],
                        generation_config={"temperature": self.temperature}
                    )
                    return (response.text or "").strip()

            except Exception as e:
                err_str = str(e)
                last_err = e
                # Exponential backoff on rate limits (429) & Quota
                if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                    # Check if daily quota hit, automatically switch to next model in pool
                    if "perday" in err_str.lower() or "limit: 20" in err_str or "quotaid" in err_str.lower():
                        if self.current_model_idx + 1 < len(self.model_pool):
                            self.current_model_idx += 1
                            self.model_name = self.model_pool[self.current_model_idx]
                            print(f"\n[Gemini API] Quota đầy, tự động chuyển sang model: {self.model_name}")
                            time.sleep(1)
                            continue

                    match = re.search(r"retry in ([0-9.]+)s", err_str, re.IGNORECASE)
                    if not match:
                        match = re.search(r"retryDelay['\":\s]+([0-9.]+)s", err_str)
                    
                    if match:
                        sleep_time = float(match.group(1)) + 3.0
                    else:
                        sleep_time = max(20.0, (2 ** attempt) * 5.0)
                    time.sleep(sleep_time)
                elif "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
                    if self.current_model_idx + 1 < len(self.model_pool):
                        self.current_model_idx += 1
                        self.model_name = self.model_pool[self.current_model_idx]
                        print(f"\n[Gemini API] Model bận/quá tải, chuyển sang: {self.model_name}")
                        time.sleep(1)
                        continue
                    time.sleep(10.0 + attempt * 5.0)
                else:
                    # 404 or other error - switch to next model
                    if "404" in err_str or "not found" in err_str.lower():
                        if self.current_model_idx + 1 < len(self.model_pool):
                            self.current_model_idx += 1
                            self.model_name = self.model_pool[self.current_model_idx]
                            print(f"\n[Gemini API] Model không hỗ trợ, chuyển sang: {self.model_name}")
                            time.sleep(1)
                            continue
                    time.sleep(2)

        raise RuntimeError(f"Lỗi gọi Gemini Vision API sau {self.max_retries} lần thử: {last_err}")

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs text recognition using Gemini Vision API on the image array.
        """
        start_time = time.perf_counter()

        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
        else:
            pil_img = Image.fromarray(image)

        full_text = self._call_gemini(pil_img)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        lines_raw = [line.strip() for line in full_text.splitlines() if line.strip()]
        lines = [{"text": line, "confidence": 0.99, "box": []} for line in lines_raw]

        return {
            "full_text": full_text,
            "lines": lines,
            "avg_confidence": 0.99 if full_text else 0.0,
            "elapse_ms": round(elapsed_ms, 2)
        }
