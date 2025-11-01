import os
from config import get_path_manager
from utils import register_datasets

def main():
    """Main function to run the prediction process."""
    pm = get_path_manager()

    # Register datasets for inference
    register_datasets(train=False, val=False, infer=True)

    # Change directory to the MaskDINO repo
    os.chdir(pm.maskdino_repo)

    # Dynamically import train_net and argument parser
    from train_net import main as maskdino_main
    from detectron2.engine import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    args = parser.parse_args()

    # Set arguments for evaluation
    args.eval_only = True
    args.num_gpus = 1

    # Get model config and weights from settings
    model_config_rel = pm.settings["maskdino"]["model_config"]
    model_weights_abs = pm._resolve_path(pm.settings["maskdino"]["model_weights"])

    args.config_file = model_config_rel
    args.opts = [
        "MODEL.WEIGHTS", str(model_weights_abs),
        "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
    ]

    print("Starting inference...")
    print(f"Config file: {pm.maskdino_repo / model_config_rel}")
    print(f"Model weights: {model_weights_abs}")

    maskdino_main(args)

if __name__ == "__main__":
    main()
