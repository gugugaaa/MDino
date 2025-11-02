import os
from config import get_path_manager
from utils import register_datasets, generate_train_config

def main():
    """Main function to run the training process."""
    pm = get_path_manager()

    # Register datasets
    register_datasets(train=True, val=True)

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

    # Set the config file path (use absolute path)
    args.config_file = str(pm.model_config)

    print(f"Starting training with config: {args.config_file}")
    maskdino_main(args)

if __name__ == "__main__":
    main()
