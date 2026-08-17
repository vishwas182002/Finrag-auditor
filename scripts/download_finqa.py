"""Download official FinQA held-out splits with pinned SHA-256 checksums."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

FILES = {
    "dev.json": (
        "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
        "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    ),
    "test.json": (
        "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
        "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
    ),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "raw"
    output.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in FILES.items():
        target = output / filename
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (pinned HTTPS)
            payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {filename}: {actual}")
        target.write_bytes(payload)
        print(f"verified {filename}: {actual}")


if __name__ == "__main__":
    main()

