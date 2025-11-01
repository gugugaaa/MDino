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

val_json = resolve_project_path(SETTINGS["datasets"]["infer_val_json"])
val_images = resolve_project_path(SETTINGS["datasets"]["infer_val_images"])

for name in ["square_val"]:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)

register_coco_instances("square_val", {}, str(val_json), str(val_images))
MetadataCatalog.get("square_val").thing_classes = ["square"]
MetadataCatalog.get("square_val").evaluator_type = "coco"

if __name__ == "__main__":
    os.chdir(maskdino_repo)

    from train_net import main
    from detectron2.engine import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)

    args = parser.parse_args()
    args.eval_only = True
    args.num_gpus = 1

    model_config_rel = Path(SETTINGS["maskdino"]["model_config"])
    model_weights_rel = Path(SETTINGS["maskdino"]["model_weights"])

    # 将相对路径转换为绝对路径
    model_weights_abs = resolve_project_path(SETTINGS["maskdino"]["model_weights"])

    args.config_file = str(model_config_rel)
    args.opts = [
        "MODEL.WEIGHTS", str(model_weights_abs),
        "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",  # 添加这行:指定类别数为1
    ]

    print("开始推理...")
    print(f"配置文件: {(maskdino_repo / model_config_rel).resolve()}")
    print(f"模型权重: {model_weights_abs}")

    main(args)