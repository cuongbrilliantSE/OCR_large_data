from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseWriter(ABC):
    @abstractmethod
    def write(self, rel_file_path: str, ocr_result: Dict[str, Any]) -> None:
        """Write OCR output for a processed file."""
        pass

    def close(self) -> None:
        """Close any open file handles."""
        pass
