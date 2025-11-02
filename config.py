import yaml
from pathlib import Path

class PathManager:
    """Manages all paths for the project."""
    def __init__(self, config_file: str = "mdino-config.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_file
        self.settings = self._load_settings()

        # MaskDINO related paths
        self.maskdino_repo = self._resolve_path(self.settings["maskdino"]["repo_path"])
        self.base_config = self._resolve_path(self.settings["maskdino"]["base_config"])
        # Model and output paths
        self.model_dir = self._resolve_path(self.settings["model"]["model_dir"])
        self.output_dir = self._resolve_path(self.settings["model"]["output_dir"])
        self.pretrained_weights = self.model_dir / self.settings["model"]["pretrained_weights"]
        self.model_weights = self.model_dir / self.settings["model"]["model_weights"]
        self.model_config = self.model_dir / self.settings["model"]["model_config"]

        # Dataset paths
        self.train_json = self._resolve_path(self.settings["datasets"]["train_json"])
        self.val_json = self._resolve_path(self.settings["datasets"]["val_json"])
        self.train_images = self._resolve_path(self.settings["datasets"]["train_images"])
        self.val_images = self._resolve_path(self.settings["datasets"]["val_images"])
        self.infer_val_json = self._resolve_path(self.settings["datasets"]["infer_val_json"])
        self.infer_val_images = self._resolve_path(self.settings["datasets"]["infer_val_images"])

    def _load_settings(self):
        """Loads settings from the YAML config file."""
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolves a path relative to the project root."""
        return (self.project_root / relative_path).resolve()

# Global instance
path_manager = PathManager()

def get_path_manager():
    """Returns the global instance of PathManager."""
    return path_manager
