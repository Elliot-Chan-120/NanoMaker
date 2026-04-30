from path_modules import *

def get_project_root() -> Path:
    """Returns absolute path to project root"""
    pack_root = Path(__file__).resolve().parent
    return pack_root.parent.parent

