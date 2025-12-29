import os
from typing import Iterable, Optional, Tuple


IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def list_image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if is_image_file(f)])


def find_image_path(folder: str, base_name: str) -> Optional[str]:
    """Return the first existing image path for base_name in folder."""
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(folder, base_name + ext)
        if os.path.exists(path):
            return path
    return None


def split_base_and_ext(filename: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(filename)
    return base, ext.lower()
