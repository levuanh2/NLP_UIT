from pathlib import Path

from scripts import download_models as module


def test_download_models_selects_component_and_destination(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)

    paths = module.download_models(tmp_path, "embedding", "test-revision")

    expected = tmp_path.resolve() / "vietnamese-legal-embedding"
    assert paths == [expected]
    assert calls == [
        {
            "repo_id": "bqbbao6/vietnamese-legal-embedding",
            "revision": "test-revision",
            "local_dir": expected,
        }
    ]


def test_download_models_downloads_all_three(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "snapshot_download",
        lambda **kwargs: calls.append(kwargs),
    )

    paths = module.download_models(tmp_path, "all", "main")

    assert len(paths) == 3
    assert len(calls) == 3
