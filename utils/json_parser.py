"""
Utility for cleaning, extracting, and parsing JSON structures from LLM responses.
"""

import json
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


class JsonParser:
    """
    Cleans raw text output from vision-language models and parses it into JSON dictionaries.
    """

    FALLBACK_SCHEMA = {
        "detected_text": [],
        "min_value": None,
        "max_value": None,
        "unit": None
    }

    @staticmethod
    def extract_json_string(raw_text: str) -> str:
        """
        Extract JSON object from model output.
        """

        if not raw_text:
            return ""

        cleaned = raw_text.strip()

        markdown_pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match_markdown = re.match(
            markdown_pattern,
            cleaned,
            re.DOTALL | re.IGNORECASE
        )

        if match_markdown:
            cleaned = match_markdown.group(1).strip()

        try:
            start_idx = cleaned.index("{")
            end_idx = cleaned.rindex("}") + 1
            cleaned = cleaned[start_idx:end_idx]
        except ValueError:
            pass

        return cleaned

    @classmethod
    def normalize_output(cls, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all expected keys exist.
        """

        normalized = cls.FALLBACK_SCHEMA.copy()

        normalized["detected_text"] = parsed_data.get(
            "detected_text",
            []
        )

        normalized["min_value"] = parsed_data.get(
            "min_value"
        )

        normalized["max_value"] = parsed_data.get(
            "max_value"
        )

        normalized["unit"] = parsed_data.get(
            "unit"
        )

        return normalized

    @classmethod
    def parse(cls, raw_text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from raw model output.
        """

        if not raw_text:
            logger.warning(
                "Empty response received. Returning fallback."
            )
            return cls.FALLBACK_SCHEMA.copy()

        json_candidate = cls.extract_json_string(raw_text)

        if not json_candidate:
            logger.warning(
                "Could not identify JSON structure in raw text: %s",
                raw_text
            )
            return cls.FALLBACK_SCHEMA.copy()

        try:
            parsed_data = json.loads(json_candidate)

            if not isinstance(parsed_data, dict):
                logger.warning(
                    "Parsed JSON is not a dictionary: %s",
                    parsed_data
                )
                return cls.FALLBACK_SCHEMA.copy()

            return cls.normalize_output(parsed_data)

        except json.JSONDecodeError as e:
            logger.error(
                "JSON decode error. Candidate string was: '%s'. Error: %s",
                json_candidate,
                e
            )

            return cls.FALLBACK_SCHEMA.copy()