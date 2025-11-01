import os
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "mdino-config.yaml"

with CONFIG_FILE.open("r", encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)


def resolve_project_path(relative_path: str) -> Path:
    return (BASE_DIR / relative_path).resolve()


maskdino_repo = resolve_project_path(SETTINGS["maskdino"]["repo_path"])
sys.path.insert(0, str(maskdino_repo))

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

train_json = resolve_project_path(SETTINGS["datasets"]["train_json"])
val_json = resolve_project_path(SETTINGS["datasets"]["val_json"])
train_images = resolve_project_path(SETTINGS["datasets"]["train_images"])
val_images = resolve_project_path(SETTINGS["datasets"]["val_images"])

for name in ["square_train", "square_val"]:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)

register_coco_instances("square_train", {}, str(train_json), str(train_images))
register_coco_instances("square_val", {}, str(val_json), str(val_images))

MetadataCatalog.get("square_train").thing_classes = ["square"]
MetadataCatalog.get("square_train").evaluator_type = "coco"
MetadataCatalog.get("square_val").thing_classes = ["square"]
MetadataCatalog.get("square_val").evaluator_type = "coco"


def write_config_yaml(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    pretrained_weights_rel = SETTINGS["maskdino"]["pretrained_weights"]
    output_dir_rel = SETTINGS["maskdino"]["output_dir"]
    output_dir_abs = (maskdino_repo / Path(output_dir_rel)).resolve()
    output_dir_abs.mkdir(parents=True, exist_ok=True)

    config_content = f"""_BASE_: "{SETTINGS["maskdino"]["base_config"]}"
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
    print(f"配置文件已生成: {config_path}")


if __name__ == "__main__":
    generated_config = BASE_DIR / "configs/square_instance.yaml"
    write_config_yaml(generated_config)

    os.chdir(maskdino_repo)

    from train_net import main  # noqa: E402
    from detectron2.engine import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)

    args = parser.parse_args()

    if not args.config_file:
        args.config_file = os.path.relpath(generated_config, maskdino_repo)

    main(args)