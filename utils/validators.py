"""
Validation utilities for parsed gauge metadata.
"""

import logging
import math
from typing import Dict, Any, Union, Optional, List

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ValidationError
)

logger = logging.getLogger(__name__)


class GaugeMetadata(BaseModel):
    """
    Validation schema for gauge metadata extraction.
    """

    detected_text: List[str] = Field(
        default_factory=list,
        description="All text detected by the model."
    )

    min_value: Optional[Union[int, float]] = Field(
        default=None,
        description="Minimum scale value."
    )

    max_value: Optional[Union[int, float]] = Field(
        default=None,
        description="Maximum scale value."
    )

    unit: Optional[str] = Field(
        default=None,
        description="Gauge measurement unit."
    )

    @field_validator("detected_text")
    @classmethod
    def validate_detected_text(
        cls,
        value: List[str]
    ) -> List[str]:

        if value is None:
            return []

        cleaned = []

        for item in value:
            if item is None:
                continue

            text = str(item).strip()

            if text:
                cleaned.append(text)

        return cleaned

    @field_validator("min_value", "max_value")
    @classmethod
    def validate_numeric(
        cls,
        value: Optional[Union[int, float]]
    ) -> Optional[Union[int, float]]:

        if value is None:
            return None

        if math.isnan(value):
            raise ValueError("NaN not allowed")

        if math.isinf(value):
            raise ValueError("Infinity not allowed")

        return value

    @field_validator("unit")
    @classmethod
    def clean_unit(
        cls,
        value: Optional[str]
    ) -> Optional[str]:

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        return value

    def validate_range(self) -> None:
        """
        Optional logical validation.
        """

        if (
            self.min_value is not None
            and self.max_value is not None
        ):
            if self.min_value >= self.max_value:
                raise ValueError(
                    f"Invalid range: "
                    f"min_value={self.min_value}, "
                    f"max_value={self.max_value}"
                )


def validate_gauge_data(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate model output.
    """

    fallback = {
        "detected_text": [],
        "min_value": None,
        "max_value": None,
        "unit": None
    }

    if not isinstance(data, dict):
        logger.error(
            "Expected dictionary but received %s",
            type(data)
        )
        return fallback

    try:

        validated = GaugeMetadata(**data)

        try:
            validated.validate_range()
        except ValueError as range_error:
            logger.warning(
                "Range validation failed: %s",
                range_error
            )

        return validated.model_dump()

    except ValidationError as validation_error:

        logger.warning(
            "Validation failed: %s",
            validation_error.errors()
        )

        return fallback

    except Exception as exception:

        logger.exception(
            "Unexpected validation error: %s",
            exception
        )

        return fallback