"""Image integrity, corruption detection, and validation utility."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, UnidentifiedImageError
import os

from ..utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageValidator:
    """Validates image files to prevent runtime crashes caused by corrupted data."""

    def __init__(self, min_size: Tuple[int, int] = (32, 32), max_file_size_mb: float = 20.0):
        self.min_size = min_size
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Check if a file exists, has a valid extension, is non-empty, and can be read by PIL.

        Returns:
            (is_valid: bool, error_reason: Optional[str])
        """
        path = Path(file_path)

        if not path.exists():
            return False, "File does not exist"

        if not path.is_file():
            return False, "Path is not a regular file"

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return False, f"Unsupported extension: {path.suffix}"

        try:
            file_size = path.stat().st_size
        except OSError as e:
            return False, f"Cannot read file stat: {e}"

        if file_size == 0:
            return False, "File is 0 bytes (empty)"

        if file_size > self.max_file_size_bytes:
            return False, f"File size exceeds {self.max_file_size_bytes / (1024 * 1024):.1f} MB limit"

        try:
            with Image.open(path) as img:
                # verify() reads headers without loading all raster data
                img.verify()

            # Re-open to verify full image data decodes without truncated bytes
            with Image.open(path) as img:
                img.load()
                w, h = img.size
                if w < self.min_size[0] or h < self.min_size[1]:
                    return False, f"Dimensions ({w}x{h}) smaller than minimum {self.min_size}"
                if img.mode not in ("RGB", "L", "RGBA"):
                    return False, f"Unsupported color mode: {img.mode}"

        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            return False, f"Corrupted or invalid image: {str(e)}"
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

        return True, None

    def scan_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> Dict[str, Union[List[Path], int]]:
        """
        Scan a directory and separate valid images from invalid/corrupted files.
        """
        dir_path = Path(directory)
        valid_files: List[Path] = []
        invalid_files: List[Dict[str, str]] = []

        pattern = "**/*" if recursive else "*"
        for item in dir_path.glob(pattern):
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                is_valid, reason = self.validate_file(item)
                if is_valid:
                    valid_files.append(item)
                else:
                    invalid_files.append({"file": str(item), "reason": reason or "Unknown"})

        return {
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "total_scanned": len(valid_files) + len(invalid_files),
            "valid_count": len(valid_files),
            "invalid_count": len(invalid_files),
        }
