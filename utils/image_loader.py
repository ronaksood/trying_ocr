"""
Utility for loading and validating gauge images.
"""

import logging
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import List, Union, Optional
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Handles image loading operations from local paths or URLs, including validation.
    """

    @staticmethod
    def load_image(image_source: Union[str, Path]) -> Image.Image:
        """
        Loads a single image from a local path or a remote URL and converts it to RGB.

        Args:
            image_source: Path to the image file or URL of the image.

        Returns:
            PIL.Image.Image: Loaded and verified image.

        Raises:
            FileNotFoundError: If a local path is provided but doesn't exist.
            ValueError: If the input is invalid or fetching from URL fails.
            UnidentifiedImageError: If PIL cannot read the image file.
        """
        # Convert path objects to string
        source_str = str(image_source)

        if source_str.startswith(("http://", "https://")):
            logger.info("Fetching remote image from URL: %s", source_str)
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                req = urllib.request.Request(source_str, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    image_bytes = response.read()
                image = Image.open(BytesIO(image_bytes))
            except Exception as e:
                logger.error("Failed to download image from URL %s: %s", source_str, e)
                raise ValueError(f"Could not load image from URL: {source_str}. Error: {e}") from e
        else:
            path = Path(image_source)
            if not path.exists():
                logger.error("Local image file not found: %s", path)
                raise FileNotFoundError(f"Image path does not exist: {path}")
            if not path.is_file():
                logger.error("Image path is not a file: %s", path)
                raise ValueError(f"Image path is not a file: {path}")
            
            logger.debug("Loading local image: %s", path)
            try:
                image = Image.open(path)
            except UnidentifiedImageError as e:
                logger.error("File is not a valid image: %s", path)
                raise e

        # Ensure image is in RGB format (handles PNG alpha channel or CMYK)
        if image.mode != "RGB":
            logger.debug("Converting image mode from %s to RGB", image.mode)
            image = image.convert("RGB")

        # Force load image data to catch corrupt file errors early
        image.load()
        return image

    @classmethod
    def load_batch(
        cls, image_sources: List[Union[str, Path]], ignore_errors: bool = False
    ) -> List[Optional[Image.Image]]:
        """
        Loads a batch of images from local paths or URLs.

        Args:
            image_sources: List of image paths or URLs.
            ignore_errors: If True, returns None for failed images instead of raising exception.

        Returns:
            List[Optional[PIL.Image.Image]]: List of loaded images. Some might be None if ignore_errors is True.
        """
        images = []
        for src in image_sources:
            try:
                img = cls.load_image(src)
                images.append(img)
            except Exception as e:
                logger.exception("Failed to load image %s", src)
                if ignore_errors:
                    images.append(None)
                else:
                    raise e
        return images
