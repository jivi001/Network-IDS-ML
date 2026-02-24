from pathlib import Path

import scripts.download_datasets as ds


def test_download_dataset_uses_urlretrieve(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_download(url: str, destination: Path):
        captured["url"] = url
        Path(destination).write_text("x", encoding="utf-8")

    monkeypatch.setattr(ds, "urlretrieve", fake_download)
    path = ds.download_dataset("nsl_kdd_sample", tmp_path)
    assert path.exists()
    assert "nsl_kdd_sample" in path.name
    assert captured["url"].startswith("http")
