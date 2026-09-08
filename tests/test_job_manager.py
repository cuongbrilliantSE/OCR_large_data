import pytest

from src.web.job_manager import OCRJobManager


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    # Reset the singleton so each test gets a clean instance.
    OCRJobManager._instance = None
    monkeypatch.chdir(tmp_path)
    return OCRJobManager(config_path="configs/config.yaml")


def test_get_job_returns_none_for_unknown_book(manager):
    assert manager.get_job("khong-ton-tai") is None


def test_get_job_returns_job_after_creation(manager):
    created = manager.get_or_create_job("bookA")
    assert manager.get_job("bookA") is created


def test_add_log_caps_at_100_entries(manager):
    for i in range(150):
        manager.add_log("bookA", f"msg {i}")
    logs = manager.get_job("bookA")["logs"]
    assert len(logs) == 100
    assert "msg 149" in logs[-1]
