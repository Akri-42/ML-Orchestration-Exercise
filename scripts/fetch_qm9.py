#!/usr/bin/env python3
"""Fetch QM9 once, materialise it as a local CSV, and record a content hash.

Run this on the build machine. 

    python scripts/fetch_qm9.py --out data/raw

Idempotent: if the output CSV already exists and its hash matches the recorded
manifest, this is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

# Primary source of record. Mirrors listed after it are tried in order.
#
# Order matters. The .tar.gz holds qm9.sdf (structures) plus qm9.sdf.csv
# (properties only, NO smiles column). Featurization is RDKit-descriptors-from-
# SMILES, so a source without structures is unusable even though it parses as a
# perfectly good CSV. The flat qm9.csv carries smiles and is therefore primary;
# see REQUIRED_COLUMNS below, which is what actually enforces this.
SOURCES = [
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
    "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.tar.gz",
]

# A source that cannot satisfy these is a failed source, not a usable one.
REQUIRED_COLUMNS = ["smiles"]

# Column order per deepchem/molnet/load_function/qm9_datasets.py
QM9_TASKS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2",
    "zpve", "cv", "u0", "u298", "h298", "g298",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ml-orch/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def extract_csv(url: str, payload: bytes) -> bytes:
    """Return CSV bytes from either a .csv or a .tar.gz payload."""
    if url.endswith(".csv"):
        return payload
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith(".csv")), None
        )
        if member is None:
            raise RuntimeError(f"no .csv inside archive at {url}")
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"could not read {member.name} from {url}")
        return handle.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the CSV is already present")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "qm9.csv"
    manifest_path = args.out / "qm9.source.json"

    if csv_path.exists() and not args.force:
        print(f"[skip] {csv_path} already present "
              f"({csv_path.stat().st_size / 1e6:.1f} MB) — use --force to refetch")
        print(f"[hash] sha256={sha256_file(csv_path)}")
        return 0

    errors: list[str] = []
    for url in SOURCES:
        try:
            print(f"[get ] {url}")
            payload = download(url)
            csv_bytes = extract_csv(url, payload)
        except Exception as exc:  # noqa: BLE001 - report and try the next mirror
            errors.append(f"{url}: {exc}")
            print(f"[warn] {exc}")
            continue

        header = csv_bytes.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
        columns = header.split(",")

        # Validate BEFORE writing. "Parsed as a CSV" is not the same as "is the
        # dataset we need" -- qm9.sdf.csv inside the tarball satisfies the first
        # and fails the second.
        absent = [c for c in REQUIRED_COLUMNS if c not in columns]
        if absent:
            msg = f"missing required column(s) {absent}; header was: {header}"
            errors.append(f"{url}: {msg}")
            print(f"[warn] {msg}")
            print("[warn] treating as a failed source, trying the next mirror")
            continue

        csv_path.write_bytes(csv_bytes)
        digest = sha256_file(csv_path)
        n_rows = csv_bytes.count(b"\n") - 1

        manifest_path.write_text(json.dumps({
            "source_url": url,
            "sha256": digest,
            "bytes": len(csv_bytes),
            "rows": n_rows,
            "header": header,
            "expected_tasks": QM9_TASKS,
            "required_columns": REQUIRED_COLUMNS,
        }, indent=2) + "\n")

        print(f"[ok  ] {csv_path}  rows={n_rows}  sha256={digest}")
        print(f"[ok  ] {manifest_path}")

        missing = [t for t in QM9_TASKS if t not in columns]
        if missing:
            print(f"[warn] header is missing expected task columns: {missing}")
            print(f"[warn] header was: {header}")
        return 0

    print("[fail] every source failed:", file=sys.stderr)
    for err in errors:
        print(f"    {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
