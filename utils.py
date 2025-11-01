import sys
import json
import numpy as np
from pathlib import Path

from config import get_path_manager

# Ensure MaskDINO is in the Python path
pm = get_path_manager()
sys.path.insert(0, str(pm.maskdino_repo))

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def register_datasets(train: bool = True, val: bool = True, infer: bool = False):
    """Registers the necessary datasets for training, validation, or inference."""
    if train:
        if "square_train" in DatasetCatalog.list():
            DatasetCatalog.remove("square_train")
        register_coco_instances("square_train", {}, str(pm.train_json), str(pm.train_images))
        MetadataCatalog.get("square_train").set(thing_classes=["square"], evaluator_type="coco")
        print("Registered 'square_train' dataset.")

    if val:
        if "square_val" in DatasetCatalog.list():
            DatasetCatalog.remove("square_val")
        register_coco_instances("square_val", {}, str(pm.val_json), str(pm.val_images))
        MetadataCatalog.get("square_val").set(thing_classes=["square"], evaluator_type="coco")
        print("Registered 'square_val' dataset.")

    if infer:
        # For inference, we often use the validation set, but point to different data
        if "square_val" in DatasetCatalog.list():
            DatasetCatalog.remove("square_val")
        register_coco_instances("square_val", {}, str(pm.infer_val_json), str(pm.infer_val_images))
        MetadataCatalog.get("square_val").set(thing_classes=["square"], evaluator_type="coco")
        print("Registered 'square_val' dataset for inference.")

def generate_train_config():
    """Generates the training config YAML file for MaskDINO."""
    config_path = pm.generated_config
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Paths relative to the MaskDINO repo
    base_config_rel = pm.settings["maskdino"]["base_config"]
    pretrained_weights_rel = pm.settings["maskdino"]["pretrained_weights"]
    output_dir_rel = str(Path(pm.settings["maskdino"]["output_dir"]).relative_to(pm.project_root.parent))

    config_content = f"""_BASE_: "{base_config_rel}"
DATASETS:
  TRAIN: ("square_train",)
  TEST: ("square_val",)
DATALOADER:
  NUM_WORKERS: 4
MODEL:
  WEIGHTS: "{pretrained_weights_rel}"
  SEM_SEG_HEAD:
    NUM_CLASSES: 1
SOLVER:
  IMS_PER_BATCH: 4
  BASE_LR: 0.0001
  MAX_ITER: 2000
  STEPS: (1500, )
  GAMMA: 0.1
  WARMUP_ITERS: 100
  AMP:
    ENABLED: True
  CHECKPOINT_PERIOD: 500
INPUT:
  MASK_FORMAT: "polygon"
  MIN_SIZE_TRAIN: (320, 352, 288)
  MAX_SIZE_TRAIN: 384
  MIN_SIZE_TEST: 320
  MAX_SIZE_TEST: 384
TEST:
  EVAL_PERIOD: 500
OUTPUT_DIR: "{output_dir_rel}"
"""
    with config_path.open("w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"Training config generated at: {config_path}")


class DataProcessor:
    """Handles loading, filtering, and processing of JSON annotation data."""
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.annotations = []
        self.filtered_annotations = []
        self.image_stats = {}

    def load_annotations(self, json_path: Path):
        """Loads annotations from a JSON file."""
        print(f"Loading annotations from {json_path}...")
        with json_path.open('r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        print(f"Loaded {len(self.annotations)} annotations.")

    def filter_by_score(self):
        """Filters annotations based on the score threshold."""
        if not self.annotations:
            print("No annotations loaded. Please call load_annotations() first.")
            return

        print(f"Filtering with threshold {self.threshold}...")
        self.filtered_annotations = [ann for ann in self.annotations if ann.get('score', 0) >= self.threshold]

        self.image_stats = {}
        for item in self.filtered_annotations:
            image_id = item.get('image_id')
            self.image_stats[image_id] = self.image_stats.get(image_id, 0) + 1

        print(f"Filtering complete. {len(self.filtered_annotations)} annotations remaining.")

    def save_annotations(self, output_path: Path, data):
        """Saves data to a JSON file."""
        print(f"Saving data to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("Save complete.")

    def print_statistics(self, data, top_k=5, title=""):
        """Prints score statistics."""
        if not data:
            print(f"{title} Statistics: No data available.")
            return

        scores = [ann.get('score', 0) for ann in data]
        sorted_scores = sorted(scores, reverse=True)[:top_k]

        print(f"--- {title} Statistics (Total: {len(scores)}) ---")
        print(f"{'Average Score':<15} {'Max Score':<15} {'Min Score':<15} {'Top {top_k} Avg':<15}")
        print(f"{np.mean(scores):<15.4f} {np.max(scores):<15.4f} {np.min(scores):<15.4f} {np.mean(sorted_scores):<15.4f}")
        print("-" * 60)

    def group_by_image(self, annotations):
        """Groups annotations by image_id."""
        image_groups = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in image_groups:
                image_groups[img_id] = []
            image_groups[img_id].append(ann)
        return image_groups
