"""
Configuration settings for the Gauge Metadata Extraction Pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure output directory exists
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0  # Greedy decoding for stable JSON structure

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Device Configuration
# Using HuggingFace's recommended setup for Qwen2.5-VL
# By default, use 'auto' to map across available GPUs, otherwise fallback
DEVICE_MAP = "auto"

# JSON validation schema configuration
REQUIRED_KEYS = {"min_value", "max_value", "unit"}
