"""Test image validation and corruption handling."""
import io
from pathlib import Path
from PIL import Image
import pytest

from src.data.validator import ImageValidator


def test_validator_detects_valid_image(tmp_path):
    validator = ImageValidator()
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    img.save(img_path, format="JPEG")

    is_valid, reason = validator.validate_file(img_path)
    assert is_valid is True
    assert reason is None


def test_validator_detects_empty_file(tmp_path):
    validator = ImageValidator()
    empty_path = tmp_path / "empty.jpg"
    empty_path.write_bytes(b"")

    is_valid, reason = validator.validate_file(empty_path)
    assert is_valid is False
    assert "empty" in reason.lower()


def test_validator_detects_corrupted_file(tmp_path):
    validator = ImageValidator()
    corrupt_path = tmp_path / "corrupt.jpg"
    # Write garbage non-image bytes
    corrupt_path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupted_garbage_bytes")

    is_valid, reason = validator.validate_file(corrupt_path)
    assert is_valid is False
    assert "corrupted" in reason.lower() or "invalid" in reason.lower()
