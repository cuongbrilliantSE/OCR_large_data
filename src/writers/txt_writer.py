from pathlib import Path
from typing import Dict, Any
from .base_writer import BaseWriter


class TxtWriter(BaseWriter):
    def __init__(self, output_dir: str | Path, mirror_structure: bool = True):
        self.output_dir = Path(output_dir)
        self.mirror_structure = mirror_structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rel_file_path: str, ocr_result: Dict[str, Any]) -> None:
        rel_path = Path(rel_file_path)
        txt_name = rel_path.with_suffix(".txt")

        if self.mirror_structure:
            target_path = self.output_dir / txt_name
        else:
            target_path = self.output_dir / rel_path.name.replace(rel_path.suffix, ".txt")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        full_text = ocr_result.get("full_text", "")
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(full_text)
