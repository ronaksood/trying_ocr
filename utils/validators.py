"""
Validation utilities for parsed gauge metadata.
"""

import logging
from typing import Dict, Any, Union, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError

logger = logging.getLogger(__name__)


class GaugeMetadata(BaseModel):
    """
    Pydantic schema for gauge metadata validation.
    """
    min_value: Optional[Union[int, float]] = Field(
        default=None,
        description="The minimum scale value printed on the gauge dial."
    )
    max_value: Optional[Union[int, float]] = Field(
        default=None,
        description="The maximum scale value printed on the gauge dial."
    )
    unit: Optional[str] = Field(
        default=None,
        description="The unit of measurement shown on the dial (e.g., 'bar', 'psi', 'C')."
    )

    @field_validator("min_value", "max_value")
    @classmethod
    def check_finite_number(cls, v: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
        """
        Verify the value is a valid finite number if present.
        """
        if v is not None:
            import math
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Value must be a finite number.")
        return v

    @field_validator("unit")
    @classmethod
    def clean_unit(cls, v: Optional[str]) -> Optional[str]:
        """
        Clean the unit string (stripping whitespace, empty to None).
        """
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                return None
            return v_stripped
        return None

    def validate_range(self) -> None:
        """
        Custom validator for cross-field range checks.
        """
        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) must be less than max_value ({self.max_value})."
                )


def validate_gauge_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates dictionary gauge data against the expected schema.
    If the data is invalid, it logs a warning and returns a fallback null JSON.

    Args:
        data: Dict containing 'min_value', 'max_value', 'unit' keys.

    Returns:
        Dict[str, Any]: Validated data.
    """
    fallback = {
        "min_value": None,
        "max_value": None,
        "unit": None
    }
    
    # Check if input is a dictionary
    if not isinstance(data, dict):
        logger.error("Data to validate is not a dictionary: %s", type(data))
        return fallback

    try:
        # Pydantic validation
        validated_model = GaugeMetadata(**data)
        # Range validation
        validated_model.validate_range()
        
        return validated_model.model_dump()
    except ValidationError as e:
        logger.warning("Schema validation failed: %s. Input data was: %s", e.errors(), data)
        return fallback
    except ValueError as e:
        logger.warning("Logical range validation failed: %s. Input data was: %s", e, data)
        return fallback
    except Exception as e:
        logger.error("Unexpected error during validation: %s", e)
        return fallback
