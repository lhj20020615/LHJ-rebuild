from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "src" / "train_multimodal.py",
        root / "src" / "data.py",
        root / "src" / "model.py",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    forbidden = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv", ".tsv", ".pt", ".pth", ".bin", ".safetensors"}
    ]
    if missing:
        raise SystemExit("Missing package files: " + ", ".join(missing))
    if forbidden:
        names = ", ".join(str(path.relative_to(root)) for path in forbidden)
        raise SystemExit("Package contains excluded data/model files: " + names)
    print(f"OK: {root}")
    print("No dataset or model-weight files are included.")


if __name__ == "__main__":
    main()

