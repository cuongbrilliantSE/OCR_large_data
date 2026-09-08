import pytest
from PIL import Image

from src.db.book_repository import BookRepository, slugify


@pytest.fixture()
def repo(tmp_path):
    return BookRepository(
        db_path=tmp_path / "tracker.db",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        covers_dir=tmp_path / "covers",
        cache_dir=tmp_path / "cache",
    )


def _make_book_with_images(repo, title, filenames):
    book_dir = repo.input_dir / title
    book_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        Image.new("RGB", (20, 20), "white").save(book_dir / name)
    repo.upsert_book(title=title, total_pages=len(filenames))
    return slugify(title)


def test_web_format_returns_source_directly(repo):
    slug = _make_book_with_images(repo, "Sach Test", ["001.jpg", "002.jpg"])
    path = repo.get_page_image_path(slug, 1)
    assert path is not None and path.name == "001.jpg"


def test_non_web_format_is_converted_and_cached(repo):
    slug = _make_book_with_images(repo, "Sach Tiff", ["001.tiff"])
    path = repo.get_page_image_path(slug, 1)
    assert path is not None
    assert path.suffix == ".jpg"
    assert path.is_file()
    assert repo.cache_dir in path.parents
    # Second call hits the cache without raising.
    assert repo.get_page_image_path(slug, 1) == path


def test_out_of_range_page_returns_none(repo):
    slug = _make_book_with_images(repo, "Sach Ngan", ["001.jpg"])
    assert repo.get_page_image_path(slug, 99) is None


def test_unknown_slug_returns_none(repo):
    assert repo.get_page_image_path("khong-co", 1) is None
