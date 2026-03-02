# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SIM10k dataset loader for cross-domain object detection (Sim10k -> Cityscapes).

Expected directory layout under DETECTRON2_DATASETS:
    sim10k/
        VOC2012/
            JPEGImages/     # *.jpg
            Annotations/    # *.xml  (PASCAL VOC format)
            ImageSets/
                Main/
                    trainval.txt   # one image stem per line

Only the "car" class is used (standard Sim10k -> Cityscapes benchmark).
"""
import logging
import os
import xml.etree.ElementTree as ET
from typing import List, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

# Standard Sim10k → Cityscapes: car only
TARGET_CLASSES = ["car"]
CLASS_NAMES = ("car",)


def load_sim10k_instances(
    data_dir: str,
    split: str = "trainval",
    class_names: Optional[List[str]] = None,
):
    """
    Load SIM10k VOC-format annotations.

    Args:
        data_dir: root of sim10k/, e.g. "datasets/sim10k"
        split: split name, reads ImageSets/Main/{split}.txt
        class_names: tuple of class names to keep

    Returns:
        list[dict]: detectron2 standard dataset dicts
    """
    if class_names is None:
        class_names = CLASS_NAMES

    class_to_id = {c: i for i, c in enumerate(class_names)}

    voc_root = os.path.join(data_dir, "VOC2012")
    image_dir = os.path.join(voc_root, "JPEGImages")
    anno_dir = os.path.join(voc_root, "Annotations")
    split_file = os.path.join(voc_root, "ImageSets", "Main", split + ".txt")

    assert PathManager.isfile(split_file), f"Split file not found: {split_file}"

    with PathManager.open(split_file, "r") as f:
        file_ids = [line.strip() for line in f if line.strip()]

    dataset_dicts = []
    for file_id in file_ids:
        xml_path = os.path.join(anno_dir, file_id + ".xml")
        image_file = os.path.join(image_dir, file_id + ".jpg")

        if not PathManager.isfile(xml_path):
            logger.warning(f"Annotation not found: {xml_path}, skipping.")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        annos = []
        for obj in root.iter("object"):
            cls_name = obj.find("name").text.strip().lower()
            if cls_name not in class_to_id:
                continue

            # Difficult flag (skip if desired; kept here for completeness)
            difficult = obj.find("difficult")
            difficult = int(difficult.text) if difficult is not None else 0

            bndbox = obj.find("bndbox")
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)

            # VOC uses 1-based coords; convert to 0-based
            x1 -= 1
            y1 -= 1
            x2 -= 1
            y2 -= 1

            if x2 <= x1 or y2 <= y1:
                continue

            anno = {
                "category_id": class_to_id[cls_name],
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "iscrowd": 0,
            }
            annos.append(anno)

        if len(annos) == 0:
            continue

        record = {
            "file_name": image_file,
            "image_id": file_id,
            "height": height,
            "width": width,
            "annotations": annos,
        }
        dataset_dicts.append(record)

    logger.info(
        f"Loaded {len(dataset_dicts)} images with annotations from SIM10k ({data_dir}, split={split})"
    )
    return dataset_dicts


def register_sim10k(
    name: str,
    data_dir: str,
    split: str = "trainval",
    class_names=CLASS_NAMES,
):
    """Register a SIM10k split with detectron2."""
    DatasetCatalog.register(
        name,
        lambda d=data_dir, s=split, c=class_names: load_sim10k_instances(
            d, split=s, class_names=c
        ),
    )
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names),
        evaluator_type="pascal_voc",
        dirname=os.path.join(data_dir, "VOC2012"),
        split=split,
        year=2012,
    )
