import threading

import pytest

from src.db.state_tracker import StateTracker


@pytest.fixture()
def tracker(tmp_path):
    return StateTracker(tmp_path / "tracker.db")


def _register(tracker, *rel_paths):
    tracker.register_files([(p, 100) for p in rel_paths])


def test_claim_marks_in_progress_and_returns_rows(tracker):
    _register(tracker, "bookA/001.jpg", "bookA/002.jpg", "bookA/003.jpg")

    claimed = tracker.claim_pending_tasks(limit=2)

    assert [c["file_path"] for c in claimed] == ["bookA/001.jpg", "bookA/002.jpg"]
    assert all("file_size" in c for c in claimed)

    stats = tracker.get_statistics()
    assert stats["in_progress"] == 2
    assert stats["pending"] == 1


def test_claim_respects_book_name_filter(tracker):
    _register(tracker, "bookA/001.jpg", "bookB/001.jpg")

    claimed = tracker.claim_pending_tasks(limit=10, book_name="bookB")

    assert [c["file_path"] for c in claimed] == ["bookB/001.jpg"]
    assert tracker.get_statistics(book_name="bookA")["pending"] == 1


def test_concurrent_claims_never_overlap(tmp_path):
    tracker = StateTracker(tmp_path / "tracker.db")
    _register(tracker, *[f"book/{i:03d}.jpg" for i in range(200)])

    seen: list[str] = []
    lock = threading.Lock()

    def worker():
        local = StateTracker(tmp_path / "tracker.db")
        while True:
            batch = local.claim_pending_tasks(limit=7)
            if not batch:
                return
            with lock:
                seen.extend(c["file_path"] for c in batch)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 200
    assert len(set(seen)) == 200
    assert tracker.get_statistics()["pending"] == 0


def test_claim_returns_empty_when_nothing_pending(tracker):
    assert tracker.claim_pending_tasks(limit=5) == []
