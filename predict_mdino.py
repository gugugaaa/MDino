import os
from config import get_path_manager
from utils import register_datasets, generate_train_config

def main():
    """Main function to run the prediction process."""
    pm = get_path_manager()

    # Register datasets for inference
    register_datasets(train=False, val=False, infer=True)

    # Generate the training config file if it doesn't exist
    if not pm.model_config.exists():
        print(f"Model config not found at {pm.model_config}. Generating...")
        generate_train_config()

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

    # Set the config file path relative to the MaskDINO repo
    args.config_file = pm.get_maskdino_relative_path(pm.model_config)
    args.opts = [
        "MODEL.WEIGHTS", str(pm.model_weights),
        "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
    ]

    print("Starting inference...")
    print(f"Config file: {pm.model_config}")
    print(f"Model weights: {pm.model_weights}")

    maskdino_main(args)

if __name__ == "__main__":
    main()
