import json
from pathlib import Path
from typing import Dict, Any
from .base_writer import BaseWriter


class JsonlWriter(BaseWriter):
    def __init__(self, output_file: str | Path):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self.output_file, "a", encoding="utf-8")

    def write(self, rel_file_path: str, ocr_result: Dict[str, Any]) -> None:
        confidence = ocr_result.get("confidence") or ocr_result.get("avg_confidence", 0.0)
        time_ms = ocr_result.get("processing_time_ms") or ocr_result.get("elapse_ms", 0.0)
        record = {
            "file_path": rel_file_path,
            "text": ocr_result.get("full_text", ""),
            "confidence": round(float(confidence), 4),
            "lines": ocr_result.get("lines", []),
            "processing_time_ms": round(float(time_ms), 2)
        }
        self._file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file_handle.flush()

    def close(self) -> None:
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()
