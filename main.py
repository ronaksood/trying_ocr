#!/usr/bin/env python
"""
Main orchestration script for the Gauge Metadata Extraction Pipeline.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

from config import settings
from models.qwen_gauge_reader import QwenGaugeReader
from utils.image_loader import ImageLoader


def setup_logging(log_file: Optional[Path] = None) -> None:
    """
    Configure global logging settings for console and optionally a file.
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        handlers=handlers
    )


def load_yolo_detections(yolo_path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads YOLO detections from a JSON file.
    Expected format:
    {
        "image_filename_or_path": [{"box_2d": [xmin, ymin, xmax, ymax], "label": "gauge"}]
    }
    """
    if not yolo_path:
        return {}
    
    path = Path(yolo_path)
    if not path.exists():
        logging.warning("YOLO detections file does not exist at: %s. Proceeding without YOLO.", yolo_path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                logging.info("Loaded YOLO detections for %d images.", len(data))
                return data
            else:
                logging.warning("YOLO detections file format invalid (expected dictionary). Proceeding without YOLO.")
                return {}
    except Exception as e:
        logging.error("Failed to read YOLO detections file: %s. Proceeding without YOLO.", e)
        return {}


def process_image(
    image_path: Path,
    reader: QwenGaugeReader,
    yolo_detections: Optional[List[Dict[str, Any]]],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Processes a single image: loads it, predicts, validates, and saves JSON.
    """
    start_time = time.time()
    logging.info("--- Processing: %s ---", image_path.name)
    
    try:
        # 1. Load image
        img = ImageLoader.load_image(image_path)
        
        # 2. Extract metadata using Qwen2.5-VL model
        result = reader.predict_single(img, yolo_detections=yolo_detections)
        
        # 3. Output results
        elapsed = time.time() - start_time
        logging.info("Extraction complete in %.2f seconds.", elapsed)
        logging.info("Result: %s", json.dumps(result))
        
        # 4. Save result to individual JSON file
        output_file = output_dir / f"{image_path.stem}_metadata.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        logging.info("Saved result to %s", output_file)
        
        return {
            "image_path": str(image_path),
            "status": "success",
            "elapsed_seconds": elapsed,
            "metadata": result
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logging.exception("Failed to process image %s", image_path)
        
        fallback_result = {
            "min_value": None,
            "max_value": None,
            "unit": None
        }
        
        # Save fallback to individual JSON file on error
        try:
            output_file = output_dir / f"{image_path.stem}_metadata.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(fallback_result, f, indent=4)
        except Exception as save_err:
            logging.error("Could not write error fallback JSON: %s", save_err)

        return {
            "image_path": str(image_path),
            "status": "error",
            "error_message": str(e),
            "elapsed_seconds": elapsed,
            "metadata": fallback_result
        }


def main() -> None:
    """
    Main entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Gauge Metadata Extraction Pipeline using Qwen2.5-VL-3B-Instruct")
    
    # Input targets
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path or URL to a single gauge image")
    input_group.add_argument("--dir", type=str, help="Path to a directory containing gauge images")
    input_group.add_argument("--images", nargs="+", type=str, help="List of paths/URLs to gauge images")
    
    # Configuration options
    parser.add_argument("--output-dir", type=str, default=str(settings.DEFAULT_OUTPUT_DIR),
                        help="Directory to save output JSON files")
    parser.add_argument("--yolo", type=str, default=None,
                        help="Path to JSON file containing YOLO detections mapping image paths to boxes")
    parser.add_argument("--model-id", type=str, default=settings.MODEL_ID,
                        help="HuggingFace model ID override")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to write log file")
    
    args = parser.parse_args()

    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = Path(args.log) if args.log else None
    setup_logging(log_file)
    
    logging.info("Starting Gauge Metadata Extraction Pipeline")
    logging.info("CUDA Available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("CUDA Device Count: %d, Current Device: %s", torch.cuda.device_count(), torch.cuda.get_device_name(0))

    # Resolve image list
    image_paths: List[Path] = []
    if args.image:
        # Single image path or url
        image_paths.append(Path(args.image))
    elif args.images:
        # Multiple images list
        image_paths.extend([Path(p) for p in args.images])
    elif args.dir:
        # Directory lookup
        dir_path = Path(args.dir)
        if not dir_path.exists():
            logging.error("Input directory does not exist: %s", dir_path)
            return
        
        # Gather standard image formats
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
        for ext in extensions:
            image_paths.extend(dir_path.glob(ext))
        
        # Keep unique values and sort
        image_paths = sorted(list(set(image_paths)))
        logging.info("Found %d images in directory: %s", len(image_paths), dir_path)

    if not image_paths:
        logging.error("No valid image sources provided or found. Exiting.")
        return

    # Load YOLO detections if specified
    yolo_detections_map = load_yolo_detections(args.yolo)

    # Initialize model (Singleton pattern - loaded once)
    try:
        reader = QwenGaugeReader(model_id=args.model_id)
    except Exception as e:
        logging.critical("Could not initialize QwenGaugeReader. Exiting. Error: %s", e)
        return

    # Execute processing loop
    results_summary = []
    pipeline_start_time = time.time()
    
    for path in image_paths:
        # Look up YOLO detections matching this path
        # Try both the absolute path string, the relative string, and just the file name
        yolo_det = None
        if yolo_detections_map:
            yolo_det = (
                yolo_detections_map.get(str(path)) or 
                yolo_detections_map.get(path.name) or 
                yolo_detections_map.get(str(path.resolve()))
            )
            if yolo_det:
                logging.info("Matched YOLO detections for image: %s", path.name)
        
        res = process_image(
            image_path=path,
            reader=reader,
            yolo_detections=yolo_det,
            output_dir=output_dir
        )
        results_summary.append(res)

    total_duration = time.time() - pipeline_start_time
    logging.info("=========================================")
    logging.info("Pipeline Execution Completed.")
    logging.info("Total processing time: %.2f seconds for %d images", total_duration, len(image_paths))
    
    # Save batch execution summary JSON
    summary_file = output_dir / "batch_summary.json"
    summary_data = {
        "timestamp": time.time(),
        "total_images": len(image_paths),
        "total_duration_seconds": total_duration,
        "results": results_summary
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
        
    logging.info("Saved batch execution summary to %s", summary_file)
    logging.info("=========================================")


if __name__ == "__main__":
    main()
