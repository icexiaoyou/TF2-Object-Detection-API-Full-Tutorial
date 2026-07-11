"""Split a flat annotated image directory into reproducible train and test directories."""
from __future__ import annotations
import argparse
import random
import shutil
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", "-i", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", "-o", type=Path)
    parser.add_argument("--test-ratio", "--ratio", "-r", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-xml", "--xml", "-x", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise SystemExit("--test-ratio must be between 0 and 1.")
    output = args.output_dir or args.image_dir
    if output == args.image_dir:
        images = [path for path in args.image_dir.iterdir() if path.suffix.lower() in EXTENSIONS]
    else:
        images = [path for path in args.image_dir.iterdir() if path.suffix.lower() in EXTENSIONS]
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    for split in ("train", "test"):
        target = output / split
        if target.exists() and any(target.iterdir()) and not args.overwrite:
            raise SystemExit(f"{target} is not empty. Use --overwrite to replace files.")
        target.mkdir(parents=True, exist_ok=True)
    random.Random(args.seed).shuffle(images)
    test_count = max(1, round(len(images) * args.test_ratio))
    for split, files in (("test", images[:test_count]), ("train", images[test_count:])):
        for image in files:
            shutil.copy2(image, output / split / image.name)
            annotation = image.with_suffix(".xml")
            if args.copy_xml and annotation.exists():
                shutil.copy2(annotation, output / split / annotation.name)
    print(f"Split {len(images)} images into {output / 'train'} and {output / 'test'}")

if __name__ == "__main__":
    main()
