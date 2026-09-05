import os
import re
from pathlib import Path
from typing import Generator, Tuple, List, Optional

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"
}


def natural_sort_key(path_or_str: str | Path):
    """
    Key for natural sorting (e.g. page_1, page_2, ..., page_9, page_10).
    Ensures pages are ordered in proper numerical sequence rather than lexicographical order.
    """
    s = str(path_or_str)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def is_image_file(path: str | Path) -> bool:
    """Check if file has an image extension."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def get_book_folders(input_dir: str | Path) -> List[Path]:
    """
    Get all direct subdirectories in input_dir (each folder represents one book).
    Sorted naturally by folder name.
    """
    base_path = Path(input_dir).resolve()
    if not base_path.exists():
        return []
    folders = [p for p in base_path.iterdir() if p.is_dir()]
    return sorted(folders, key=lambda p: natural_sort_key(p.name))


def get_book_pages(book_dir: str | Path) -> List[Path]:
    """
    Get all image files inside a specific book folder (recursively),
    sorted naturally by file path to preserve book page order.
    """
    book_path = Path(book_dir).resolve()
    if not book_path.exists():
        return []
    images = [
        p for p in book_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: natural_sort_key(p.as_posix()))


def scan_image_files(
    input_dir: str | Path,
    book_name: Optional[str] = None
) -> Generator[Tuple[str, int], None, None]:
    """
    Scan input_dir for valid image files, organized by book folders.
    If book_name is specified, scan only that book's folder.
    Images within each book are yielded in natural reading order.
    Yields tuple of (relative_path_str, file_size_in_bytes).
    """
    base_path = Path(input_dir).resolve()
    if not base_path.exists():
        return

    if book_name:
        target_dir = base_path / book_name
        if target_dir.is_dir():
            for img in get_book_pages(target_dir):
                try:
                    rel_path = img.relative_to(base_path).as_posix()
                    yield rel_path, img.stat().st_size
                except (OSError, ValueError):
                    continue
        return

    # 1. First scan through book folders in natural order
    book_folders = get_book_folders(base_path)
    for folder in book_folders:
        for img in get_book_pages(folder):
            try:
                rel_path = img.relative_to(base_path).as_posix()
                yield rel_path, img.stat().st_size
            except (OSError, ValueError):
                continue

    # 2. Also yield any loose images at root (if any)
    root_images = [
        f for f in base_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    for img in sorted(root_images, key=lambda p: natural_sort_key(p.name)):
        try:
            rel_path = img.relative_to(base_path).as_posix()
            yield rel_path, img.stat().st_size
        except (OSError, ValueError):
            continue

