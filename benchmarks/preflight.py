"""Read-only GPU environment check before a real NanoQoS benchmark."""

import argparse
import importlib.util
import json
import platform
import re
from importlib import metadata
from pathlib import Path


def inspect_model(model: Path) -> dict[str, bool]:
    if not model.is_dir():
        return {"model_directory": False, "model_config": False,
                "tokenizer": False, "weights": False}
    files = {path.name for path in model.iterdir() if path.is_file()}
    return {
        "model_directory": True,
        "model_config": "config.json" in files,
        "tokenizer": "tokenizer.json" in files or "tokenizer.model" in files,
        "weights": any(name.endswith(".safetensors") or name.endswith(".bin")
                       for name in files),
    }


def inspect_environment(model: Path, attention_backend: str = "flash") -> dict:
    if attention_backend not in {"flash", "sdpa"}:
        raise ValueError("attention_backend must be flash or sdpa")
    checks = {"linux_or_wsl": platform.system() == "Linux"}
    checks.update(inspect_model(model))
    versions = {}
    for distribution, module in (("triton", "triton"),
                                 ("flash-attn", "flash_attn"),
                                 ("transformers", "transformers")):
        try:
            versions[distribution] = metadata.version(distribution)
            checks[distribution] = importlib.util.find_spec(module) is not None
        except metadata.PackageNotFoundError:
            checks[distribution] = False
    transformers_version = re.match(r"^(\d+)\.(\d+)", versions.get("transformers", ""))
    checks["transformers_min_4_51"] = bool(
        transformers_version and tuple(map(int, transformers_version.groups())) >= (4, 51))
    gpu = {}
    try:
        import torch
        versions["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["nccl_available"] = torch.distributed.is_nccl_available()
        if checks["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            gpu = {"name": props.name, "total_memory_bytes": props.total_memory,
                   "compute_capability": list(torch.cuda.get_device_capability(0)),
                   "cuda_runtime": torch.version.cuda}
    except (ImportError, OSError):
        checks["cuda_available"] = False
        checks["nccl_available"] = False
    required = [name for name in checks if attention_backend == "flash" or name != "flash-attn"]
    return {"ready": all(checks[name] for name in required), "checks": checks,
            "required_checks": required, "attention_backend": attention_backend,
            "versions": versions, "gpu": gpu,
            "model": str(model.resolve()), "platform": platform.platform()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("Qwen"))
    parser.add_argument("--attention-backend", choices=["flash", "sdpa"], default="flash")
    args = parser.parse_args(argv)
    report = inspect_environment(args.model, args.attention_backend)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
