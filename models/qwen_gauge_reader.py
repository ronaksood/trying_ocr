"""
Model wrapper for Qwen2.5-VL-3B-Instruct for analog gauge metadata extraction.
Uses a thread-safe Singleton pattern to ensure the model is loaded only once.
"""

import logging
import threading
from typing import Dict, Any, List, Union, Optional, Tuple
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import settings
from prompts.gauge_prompt import SYSTEM_PROMPT, USER_PROMPT
from utils.json_parser import JsonParser
from utils.validators import validate_gauge_data

logger = logging.getLogger(__name__)


class QwenGaugeReader:
    """
    Singleton class to manage Qwen2.5-VL-3B-Instruct loading and inference.
    """
    _instance: Optional["QwenGaugeReader"] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "QwenGaugeReader":
        """
        Ensure thread-safe Singleton creation.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QwenGaugeReader, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = settings.MODEL_ID, device_map: str = settings.DEVICE_MAP) -> None:
        """
        Initializes the model and processor once.
        """
        if self._initialized:
            return

        logger.info("Initializing QwenGaugeReader instance...")
        self.model_id = model_id
        self.device_map = device_map

        # Determine dtype based on CUDA capability
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            self.torch_dtype = torch.float32

        logger.info("Loading model %s on device_map=%s with dtype=%s...", self.model_id, self.device_map, self.torch_dtype)
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map
            )
            self.model.eval()  # Load in inference mode only
            
            logger.info("Loading processor for %s...", self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self._initialized = True
            logger.info("QwenGaugeReader loaded successfully.")
        except Exception as e:
            logger.critical("Failed to load Qwen2.5-VL model or processor: %s", e)
            raise e

    @torch.no_grad()
    def predict_single(
        self, 
        image: Image.Image, 
        yolo_detections: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extract gauge metadata from a single PIL image.

        Args:
            image: PIL.Image object.
            yolo_detections: Optional future YOLO detections containing bounding boxes.
                             Format: [{"box_2d": [xmin, ymin, xmax, ymax], "label": "gauge"}]
                             If provided, the image is cropped to the detected gauge region
                             before running vision model inference.

        Returns:
            Dict[str, Any]: Validated JSON object containing min_value, max_value, and unit.
        """
        logger.info("Running prediction on a single image.")
        processed_image = image

        # Future Compatibility: Crop to YOLO detection if available
        if yolo_detections and len(yolo_detections) > 0:
            logger.info("YOLO detections provided. Processing first detected gauge box.")
            detection = yolo_detections[0]
            box = detection.get("box_2d")
            if box and len(box) == 4:
                # Bounding box format: [xmin, ymin, xmax, ymax]
                # Ensure box has integer coordinates
                xmin, ymin, xmax, ymax = map(int, box)
                logger.info("Cropping image to box: [%d, %d, %d, %d]", xmin, ymin, xmax, ymax)
                try:
                    processed_image = image.crop((xmin, ymin, xmax, ymax))
                except Exception as e:
                    logger.error("Failed to crop image using YOLO detections: %s. Using full image.", e)
            else:
                logger.warning("Invalid bounding box format in yolo_detections. Using full image.")

        # Construct messages template
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": USER_PROMPT}
                ]
            }
        ]

        try:
            # Prepare texts and visual inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Generate output
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=settings.DEFAULT_MAX_NEW_TOKENS,
                temperature=settings.DEFAULT_TEMPERATURE,
                do_sample=False
            )
            
            # Trim output to get response only
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.debug("Raw model output: %s", output_text)
            
            # Parse & validate response
            raw_json = JsonParser.parse(output_text)
            validated_json = validate_gauge_data(raw_json)
            return validated_json

        except Exception as e:
            logger.error("Exception during model inference or extraction: %s", e)
            return {"min_value": None, "max_value": None, "unit": None}

    @torch.no_grad()
    def predict_batch(
        self, 
        images: List[Image.Image], 
        yolo_detections_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract gauge metadata from a batch of images.

        Args:
            images: List of PIL.Image objects.
            yolo_detections_list: Optional list of YOLO detections corresponding to each image.

        Returns:
            List[Dict[str, Any]]: List of validated JSON dicts for each input image.
        """
        logger.info("Running prediction on a batch of %d images.", len(images))
        results = []

        # We process batches sequentially or in true parallel depending on hardware capability.
        # To maintain high reliability and support various GPUs, we iterate through images,
        # but utilize the singleton model instance.
        for idx, img in enumerate(images):
            yolo_det = None
            if yolo_detections_list and idx < len(yolo_detections_list):
                yolo_det = yolo_detections_list[idx]
            
            logger.info("Processing batch image %d/%d...", idx + 1, len(images))
            res = self.predict_single(img, yolo_detections=yolo_det)
            results.append(res)
            
        return results
