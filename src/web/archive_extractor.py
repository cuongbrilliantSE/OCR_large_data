import os
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Tuple, List

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}

def extract_archive(archive_path: Path, input_root: Path, custom_book_name: str = None) -> Tuple[str, Path, int]:
    """
    Extracts an uploaded archive (.rar, .zip, .7z, etc.) into input_root/<book_name>.
    Normalizes nested folders so images are located directly under the book directory.
    Returns:
        (book_name, target_dir, image_count)
    """
    archive_path = Path(archive_path).resolve()
    input_root = Path(input_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)

    # Determine book name from file stem if not custom
    import re
    raw_name = custom_book_name.strip() if custom_book_name else archive_path.stem
    # Strip Google drive timestamps e.g. -20260906T025346Z-1-001
    raw_name = re.sub(r"-\d{8}T\d{6}Z.*$", "", raw_name)
    book_name = "".join(c for c in raw_name if c not in '<>:"/\\|?*').strip() or "Untitled_Book"

    target_dir = input_root / book_name
    temp_extract_dir = input_root / f".tmp_{archive_path.stem}_{os.getpid()}"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Determine extraction method
        ext = archive_path.suffix.lower()
        extracted = False

        # Try 7zz bundled binary or system 7z/7zz for RAR/7Z/all archives
        binary_7z = Path(__file__).resolve().parent.parent.parent / "bin" / "7zz"
        executable_7z = None
        if binary_7z.exists() and os.access(binary_7z, os.X_OK):
            executable_7z = str(binary_7z)
        elif shutil.which("7zz"):
            executable_7z = shutil.which("7zz")
        elif shutil.which("7z"):
            executable_7z = shutil.which("7z")

        if executable_7z:
            cmd = [executable_7z, "x", "-y", f"-o{temp_extract_dir}", str(archive_path)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                extracted = True


        # Fallback to python standard zipfile for .zip
        if not extracted and ext == ".zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            extracted = True

        if not extracted:
            raise RuntimeError(f"Không thể giải nén file {archive_path.name}. Hãy đảm bảo file nén đúng định dạng .rar, .zip hoặc .7z.")

        # Find all images extracted
        all_files = [p for p in temp_extract_dir.rglob("*") if p.is_file()]
        image_files = [p for p in all_files if p.suffix.lower() in VALID_IMAGE_EXTS]

        if not image_files:
            raise ValueError(f"File nén '{archive_path.name}' không chứa file ảnh hợp lệ nào ({', '.join(VALID_IMAGE_EXTS)}).")

        # Create target book folder
        target_dir.mkdir(parents=True, exist_ok=True)

        # Move images directly into target_dir (flattening nested directory structures if any)
        copied_count = 0
        used_names = set()
        for img in image_files:
            dest_name = img.name
            # Avoid collision if multiple nested subfolders have same file name
            if dest_name in used_names:
                dest_name = f"{img.parent.name}_{img.name}"
            used_names.add(dest_name)
            
            dest_path = target_dir / dest_name
            shutil.copy2(img, dest_path)
            copied_count += 1

        return book_name, target_dir, copied_count

    finally:
        # Clean up temporary extract folder
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
