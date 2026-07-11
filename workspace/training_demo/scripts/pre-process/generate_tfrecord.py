"""Convert Pascal VOC XML annotations to a TFRecord using TensorFlow 2 APIs."""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", "-x", type=Path, required=True)
    parser.add_argument("--labels-path", "-l", type=Path, required=True)
    parser.add_argument("--output-path", "-o", type=Path, required=True)
    parser.add_argument("--image-dir", "-i", type=Path)
    return parser.parse_args()

def make_example(xml_path: Path, image_dir: Path, label_ids: dict[str, int]) -> tf.train.Example:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    image_path = image_dir / filename if filename else xml_path.with_suffix(".jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"{xml_path}: image not found: {image_path}")
    encoded = tf.io.gfile.GFile(str(image_path), "rb").read()
    width, height = Image.open(io.BytesIO(encoded)).size
    xmins, xmaxs, ymins, ymaxs, classes, classes_text = [], [], [], [], [], []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in label_ids:
            raise ValueError(f"{xml_path}: {name!r} is not in {label_ids}")
        box = obj.find("bndbox")
        xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
        xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
        xmins.append(max(0, xmin) / width); xmaxs.append(min(width, xmax) / width)
        ymins.append(max(0, ymin) / height); ymaxs.append(min(height, ymax) / height)
        classes.append(label_ids[name]); classes_text.append(name.encode("utf-8"))
    features = {
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(image_path.name.encode("utf-8")),
        "image/source_id": dataset_util.bytes_feature(image_path.name.encode("utf-8")),
        "image/encoded": dataset_util.bytes_feature(encoded),
        "image/format": dataset_util.bytes_feature(image_path.suffix.lstrip(".").lower().encode("utf-8")),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
        "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_util.int64_list_feature(classes),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

def main() -> None:
    args = parse_args()
    label_map = label_map_util.load_labelmap(str(args.labels_path))
    label_ids = label_map_util.get_label_map_dict(label_map)
    image_dir = args.image_dir or args.xml_dir
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(args.output_path)) as writer:
        for xml_path in sorted(args.xml_dir.glob("*.xml")):
            writer.write(make_example(xml_path, image_dir, label_ids).SerializeToString())
    print(f"Created {args.output_path}")

if __name__ == "__main__":
    main()
