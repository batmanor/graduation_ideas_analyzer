import pytest

from app.utils import download_model


def _configure_model_paths(monkeypatch, model_dir):
    monkeypatch.setattr(download_model.settings, "EMBEDDING_MODEL_PATH", str(model_dir))
    monkeypatch.setattr(download_model.settings, "EMBEDDING_ONNX_FILE", "model.onnx")
    monkeypatch.setattr(
        download_model.settings, "EMBEDDING_POOLING_CONFIG_PATH", "1_Pooling"
    )


def _write_required_model_files(model_dir):
    (model_dir / "1_Pooling").mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"onnx")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")


def test_ensure_model_skips_download_when_model_files_exist(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_required_model_files(model_dir)
    _configure_model_paths(monkeypatch, model_dir)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def fail_download(**_kwargs):
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setattr(download_model, "snapshot_download", fail_download)

    download_model.get_model()


def test_ensure_model_rejects_missing_model_when_huggingface_is_offline(
    tmp_path, monkeypatch
):
    _configure_model_paths(monkeypatch, tmp_path / "missing-model")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"):
        download_model.get_model()


def test_ensure_model_downloads_missing_model_when_online(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _configure_model_paths(monkeypatch, model_dir)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def fake_download(**kwargs):
        assert kwargs["repo_id"] == download_model.settings.REPO_ID
        assert kwargs["local_dir"] == model_dir
        _write_required_model_files(model_dir)

    monkeypatch.setattr(download_model, "snapshot_download", fake_download)

    download_model.get_model()
