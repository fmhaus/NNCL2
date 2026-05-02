"""Extract ImageNet-100 from ILSVRC2012 tar files.

Selectively extracts only the 100 needed classes — no need to unpack the full 137 GB.

Storage needed: ~13 GB

Usage:
  uv run python create_imagenet100.py \
    --dataset-path /path/to/ilsvrc2012 \
    --output       ./imagenet100
"""

import argparse
import io
import struct
import tarfile
from pathlib import Path

CLASSES_FILE = Path(__file__).parent / "solo/data/dataset_subset/imagenet100_classes.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-path", required=True,
                   help="Directory containing the ILSVRC2012 tar files.")
    p.add_argument("--output", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Devkit parsing — extracts (val_index → wnid) mapping without scipy
# ---------------------------------------------------------------------------

def _parse_devkit(devkit_path: str) -> dict:
    """Return {1-based val image index: wnid} for all 50000 val images."""
    import re

    with tarfile.open(devkit_path, "r:gz") as outer:
        # ground truth: ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
        gt_member = next(
            m for m in outer.getmembers()
            if m.name.endswith("ILSVRC2012_validation_ground_truth.txt")
        )
        gt_text = outer.extractfile(gt_member).read().decode()
        # one class-index (1-1000) per line, in val image order
        class_indices = [int(x) for x in gt_text.split()]

        # meta.mat contains the synset info; parse wnids without scipy using raw bytes
        meta_member = next(m for m in outer.getmembers() if m.name.endswith("meta.mat"))
        meta_bytes   = outer.extractfile(meta_member).read()

    # Extract all wnid strings (nXXXXXXXX) from the binary .mat in order
    wnids = re.findall(rb"(n\d{8})", meta_bytes)
    # They appear paired: first occurrence is the wnid for class index 1..1000
    wnids = [w.decode() for w in wnids]
    # .mat stores wnids in synset order; deduplicate while preserving order
    seen, unique_wnids = set(), []
    for w in wnids:
        if w not in seen:
            seen.add(w)
            unique_wnids.append(w)
    # unique_wnids[i] corresponds to class index i+1
    idx_to_wnid = {i + 1: w for i, w in enumerate(unique_wnids)}

    return {i + 1: idx_to_wnid[cls_idx] for i, cls_idx in enumerate(class_indices)}


# ---------------------------------------------------------------------------
# Train extraction — selectively unpack only the 100 class tars
# ---------------------------------------------------------------------------

def extract_train(train_tar: str, output_dir: Path, classes: set) -> None:
    print(f"Opening train tar: {train_tar}")
    with tarfile.open(train_tar, "r:") as outer:
        members = [m for m in outer.getmembers() if Path(m.name).stem in classes]
        print(f"Found {len(members)}/{len(classes)} class tars in train archive.")

        for i, member in enumerate(members, 1):
            wnid     = Path(member.name).stem
            class_dir = output_dir / "train" / wnid
            class_dir.mkdir(parents=True, exist_ok=True)

            fobj = outer.extractfile(member)
            with tarfile.open(fileobj=fobj) as inner:
                inner.extractall(class_dir)

            print(f"  [{i:3d}/{len(members)}] {wnid} ({len(list(class_dir.iterdir()))} images)")


# ---------------------------------------------------------------------------
# Val extraction — extract all, keep only the 100 classes
# ---------------------------------------------------------------------------

def extract_val(val_tar: str, devkit: str, output_dir: Path, classes: set) -> None:
    print(f"\nParsing devkit for val labels: {devkit}")
    val_to_wnid = _parse_devkit(devkit)

    print(f"Opening val tar: {val_tar}")
    kept = 0
    with tarfile.open(val_tar, "r:") as tf:
        members = sorted(tf.getmembers(), key=lambda m: m.name)
        for i, member in enumerate(members, 1):
            wnid = val_to_wnid.get(i)
            if wnid not in classes:
                continue
            class_dir = output_dir / "val" / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            fobj = tf.extractfile(member)
            (class_dir / member.name).write_bytes(fobj.read())
            kept += 1

    print(f"  Kept {kept} val images across {len(classes)} classes.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    src     = Path(args.dataset_path)
    classes = set(CLASSES_FILE.read_text().split())
    output  = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_tar = src / "ILSVRC2012_img_train.tar"
    val_tar   = src / "ILSVRC2012_img_val.tar"
    devkit    = src / "ILSVRC2012_devkit_t12.tar.gz"

    for f in (train_tar, val_tar, devkit):
        if not f.exists():
            raise SystemExit(f"File not found: {f}")

    print(f"Source : {src}")
    print(f"Output : {output}")
    print(f"Classes: {len(classes)}\n")

    extract_train(str(train_tar), output, classes)
    extract_val(str(val_tar), str(devkit), output, classes)

    print(f"\nDone. Dataset at: {output}")
    print(f"  train: {sum(1 for _ in (output/'train').rglob('*.JPEG'))} images")
    print(f"  val:   {sum(1 for _ in (output/'val').rglob('*.JPEG'))} images")


if __name__ == "__main__":
    main()
