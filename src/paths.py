from pathlib import Path

def get_project_root() -> Path:
    """Returns absolute path to nano_maker"""
    pack_root = Path(__file__).resolve().parent
    return pack_root.parent.parent

PROJECT_ROOT = get_project_root()
SOURCE_ROOT = PROJECT_ROOT / "src"

DATABASE = SOURCE_ROOT / "database"

NANOMAKER_ROOT = SOURCE_ROOT / "nano_maker"
NANO_CONTAINER = NANOMAKER_ROOT / "container"
NANO_MODULES = NANOMAKER_ROOT / "modules"
NANO_UTILITY = NANOMAKER_ROOT / "utility"