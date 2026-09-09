"""Fix HuggingFace parquet schema metadata for LeRobot v2.1 datasets.

Some converters write ``_type: \"List\"`` for vector features. Newer ``datasets`` rejects
``List`` and expects ``Sequence``, which breaks ``LeRobotDataset`` / ``load_dataset('parquet', ...)``.

Usage:
  uv run scripts/fix_lerobot_parquet_hf_metadata.py /path/to/dataset_root
  # dataset_root: folder that contains ``data/`` and ``meta/`` (e.g. ``.../data_converted``).

Optional: ``--dry-run`` to only print files that would change.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import pyarrow.parquet as pq
import tyro


def _count_list_markers(obj: Any) -> int:
    n = 0
    if isinstance(obj, dict):
        if obj.get("_type") == "List":
            n += 1
        for v in obj.values():
            n += _count_list_markers(v)
    elif isinstance(obj, list):
        for x in obj:
            n += _count_list_markers(x)
    return n


def _patch_hf_metadata(obj: Any) -> int:
    """Recursively replace ``_type: List`` -> ``Sequence``. Returns number of replacements."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("_type") == "List":
            obj["_type"] = "Sequence"
            n += 1
        for v in obj.values():
            n += _patch_hf_metadata(v)
    elif isinstance(obj, list):
        for x in obj:
            n += _patch_hf_metadata(x)
    return n


def fix_parquet(path: pathlib.Path) -> tuple[bool, int]:
    """Return (changed, num_list_replacements)."""
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    if b"huggingface" not in meta:
        return False, 0
    payload = json.loads(meta[b"huggingface"])
    n = _patch_hf_metadata(payload)
    if n == 0:
        return False, 0
    meta[b"huggingface"] = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    out = table.replace_schema_metadata(meta)
    pq.write_table(out, path)
    return True, n


def main(
    dataset_root: pathlib.Path,
    *,
    dry_run: bool = False,
):
    """Patch every ``*.parquet`` under ``dataset_root/data`` (or under ``dataset_root`` if no data/)."""
    data_dir = dataset_root / "data" if (dataset_root / "data").is_dir() else dataset_root
    paths = sorted(data_dir.rglob("*.parquet"))
    if not paths:
        print(f"No parquet files under {data_dir}", file=sys.stderr)
        raise SystemExit(1)

    changed = 0
    total_lists = 0
    if dry_run:
        for p in paths:
            t = pq.read_table(p)
            meta = dict(t.schema.metadata or {})
            if b"huggingface" not in meta:
                continue
            payload = json.loads(meta[b"huggingface"])
            n = _count_list_markers(payload)
            if n:
                print(f"would patch {p} ({n} List markers)")
                changed += 1
                total_lists += n
        print(f"\nDry run: {changed} files would be modified ({total_lists} List markers).")
        return

    for p in paths:
        did, n = fix_parquet(p)
        if did:
            print(f"patched {p} (List->Sequence: {n})")
            changed += 1
            total_lists += n

    print(f"\nDone: modified {changed} parquet files ({total_lists} List->Sequence replacements).")


if __name__ == "__main__":
    tyro.cli(main)
