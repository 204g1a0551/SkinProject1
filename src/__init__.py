"""
AI Skin: Intelligent Skin Diseases Detection
Core machine learning, explainability, and evaluation package.
"""
import os
import tempfile

# Prevent matplotlib cache permission warning
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

__version__ = "0.1.0"
