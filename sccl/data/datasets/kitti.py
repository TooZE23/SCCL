# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
KITTI dataset loader for cross-domain object detection.

Expected directory layout under DETECTRON2_DATASETS:
    kitti/
        training/
            image_2/        # 000000.png, 000001.png, ...
            label_2/        # 000000.txt, 000001.txt, ...

Each label_2 txt file contains one object per line:
    type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
We only keep objects whose type is in TARGET_CLASSES (default: ["Car"]).
"""
import logging
import os
import numpy as np
from typing import List, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

# Classes to keep for the KITTI → Cityscapes DA benchmark (car only)
TARGET_CLASSES = ["Car", "Van"]  # both mapped to "car"
CLASS_NAMES = ("car",)


def load_kitti_instances(
    data_dir: str,
    split: str = "train",
    class_names: Optional[List[str]] = None,
):
    """
    Load KITTI object detection annotations.

    Args:
        data_dir: root of kitti/, e.g. "datasets/kitti"
        split: "train" (uses training/) or a custom split file
        class_names: tuple of class names used for this dataset

    Returns:
        list[dict]: detectron2 standard dataset dicts
    """
    if class_names is None:
        class_names = CLASS_NAMES

    class_to_id = {c: i for i, c in enumerate(class_names)}
    image_dir = os.path.join(data_dir, "training", "image_2")
    label_dir = os.path.join(data_dir, "training", "label_2")

    assert PathManager.isdir(image_dir), f"Image dir not found: {image_dir}"
    assert PathManager.isdir(label_dir), f"Label dir not found: {label_dir}"

    # Collect all label files and sort for reproducibility
    label_files = sorted(
        [f for f in PathManager.ls(label_dir) if f.endswith(".txt")]
    )

    dataset_dicts = []
    for label_file in label_files:
        stem = os.path.splitext(label_file)[0]
        image_file = os.path.join(image_dir, stem + ".png")
        if not PathManager.isfile(image_file):
            # Try .jpg as fallback
            image_file = os.path.join(image_dir, stem + ".jpg")
            if not PathManager.isfile(image_file):
                logger.warning(f"Image not found for label {label_file}, skipping.")
                continue

        label_path = os.path.join(label_dir, label_file)
        annos = _parse_kitti_label(label_path, class_to_id)

        # Skip images with no valid annotations
        if len(annos) == 0:
            continue

        record = {
            "file_name": image_file,
            "image_id": stem,
            "height": None,  # will be filled lazily by detectron2
            "width": None,
            "annotations": annos,
        }
        # Read image size (needed by detectron2)
        # Lazy approach: read from PIL only once
        from PIL import Image

        with PathManager.open(image_file, "rb") as f:
            img = Image.open(f)
            record["width"], record["height"] = img.size

        dataset_dicts.append(record)

    logger.info(
        f"Loaded {len(dataset_dicts)} images with annotations from KITTI ({data_dir})"
    )
    return dataset_dicts


def _parse_kitti_label(label_path: str, class_to_id: dict):
    """
    Parse a single KITTI label_2 txt file.

    Each line: type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
    Returns list of annotation dicts in detectron2 format.
    """
    annos = []
    with PathManager.open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj_type = parts[0]

            # Map "Car" and "Van" to "car"
            if obj_type in TARGET_CLASSES:
                cat_id = class_to_id.get("car", None)
            else:
                continue  # skip Pedestrian, Cyclist, DontCare, etc.

            if cat_id is None:
                continue

            # 2D bounding box (0-indexed, pixel coords)
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])

            # Basic validity check
            if x2 <= x1 or y2 <= y1:
                continue

            truncated = float(parts[1])
            occluded = int(parts[2])

            # Skip heavily truncated / occluded boxes (optional, common practice)
            # DontCare objects are already filtered by class name
            anno = {
                "category_id": cat_id,
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "truncated": truncated,
                "occluded": occluded,
            }
            annos.append(anno)
    return annos


def register_kitti(name: str, data_dir: str, class_names=CLASS_NAMES):
    """Register a KITTI split with detectron2."""
    DatasetCatalog.register(
        name,
        lambda d=data_dir, c=class_names: load_kitti_instances(d, class_names=c),
    )
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names),
        evaluator_type="coco",
    )
