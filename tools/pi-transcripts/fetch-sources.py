"""Fetch pinned source artifacts into an ignored directory, without model access."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen


def verify(path: Path, reference: dict) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != reference["sha256"] or ("bytes" in reference and path.stat().st_size != reference["bytes"]):
        raise ValueError(f"artifact hash mismatch or size mismatch: {path}")


def download(reference: dict, path: Path) -> None:
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        with urlopen(reference["url"], timeout=120) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.rename(path)
    verify(path, reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--context-only", action="store_true", help="history/metadata only; omits workspace checkpoints")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_bytes())
    selected = [item for item in catalog["artifacts"] if args.cases is None or item["scenario"] in args.cases]
    if not selected or (args.cases and set(args.cases) - {item["scenario"] for item in selected}):
        parser.error("unknown or empty case selection")
    args.output.mkdir(parents=True, exist_ok=True)
    for item in selected:
        if Path(item["scenario"]).name != item["scenario"] or item["scenario"] in {"", ".", ".."}:
            raise ValueError("invalid case directory name")
        destination = args.output / item["scenario"]
        if destination.exists():
            raise ValueError(f"refusing to overwrite {destination}")
        reference = item.get("contextOnlyBundle") if args.context_only else item
        if not reference:
            raise ValueError(f"no context-only bundle published: {item['scenario']}")
        suffix = ".context.tar.gz" if args.context_only else ".source.tar.gz"
        archive = args.output / (item["scenario"] + suffix)
        if parts := reference.get("parts"):
            if not archive.exists():
                partial = archive.with_suffix(archive.suffix + ".partial")
                with partial.open("xb") as output:
                    for index, part_reference in enumerate(parts, 1):
                        part = args.output / f"{item['scenario']}.source.part-{index:03d}"
                        download(part_reference, part)
                        with part.open("rb") as stream:
                            shutil.copyfileobj(stream, output, length=1024 * 1024)
                partial.rename(archive)
            verify(archive, reference)
        else:
            download(reference, archive)
        # The outer bundle contains regular metadata/history/checkpoint archives;
        # runtime checkpoint restoration has its own stricter task-root extractor.
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise ValueError("outer source bundles cannot contain links or special files")
                if Path(member.name).parts[0] != item["scenario"]:
                    raise ValueError("outer source bundle has an unexpected root")
            source.extractall(args.output, filter="data")
        print(json.dumps({"scenario": item["scenario"], "sha256": reference["sha256"], "directory": str(destination),
                          "contextOnly": args.context_only}))


if __name__ == "__main__":
    main()
