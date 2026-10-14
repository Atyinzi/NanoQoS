from benchmarks.preflight import inspect_environment, inspect_model


def test_preflight_requires_model_files(tmp_path):
    missing = inspect_model(tmp_path / "missing")
    assert not any(missing.values())
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"test")
    assert all(inspect_model(tmp_path).values())


def test_sdpa_preflight_marks_flash_attention_optional(tmp_path):
    report = inspect_environment(tmp_path, attention_backend="sdpa")
    assert report["attention_backend"] == "sdpa"
    assert "flash-attn" not in report["required_checks"]
