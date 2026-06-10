from pathlib import Path

def get_project_root() -> Path:
    """Returns absolute path to nano_maker"""
    pack_root = Path(__file__).resolve().parent
    return pack_root.parent

PROJECT_ROOT = get_project_root()
SOURCE_ROOT = PROJECT_ROOT / "src"
DATABASE = PROJECT_ROOT / "database"

nnmkr = SOURCE_ROOT / "nano_maker"
CONTAINER = nnmkr / "container"
POCKET_DATA = SOURCE_ROOT / "output_container"