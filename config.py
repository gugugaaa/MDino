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

        # Dataset paths
        self.train_json = self._resolve_path(self.settings["datasets"]["train_json"])
        self.val_json = self._resolve_path(self.settings["datasets"]["val_json"])
        self.train_images = self._resolve_path(self.settings["datasets"]["train_images"])
        self.val_images = self._resolve_path(self.settings["datasets"]["val_images"])
        self.infer_val_json = self._resolve_path(self.settings["datasets"]["infer_val_json"])
        self.infer_val_images = self._resolve_path(self.settings["datasets"]["infer_val_images"])

        # Model and output paths
        self.output_dir = self._resolve_path(self.settings["maskdino"]["output_dir"])
        self.generated_config = self.project_root / "configs/square_instance.yaml"

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

    def get_maskdino_relative_path(self, abs_path: Path) -> str:
        """Returns the path relative to the MaskDINO repo."""
        return str(abs_path.relative_to(self.maskdino_repo))

# Global instance
path_manager = PathManager()

def get_path_manager():
    """Returns the global instance of PathManager."""
    return path_manager
