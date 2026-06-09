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

    @staticmethod
    def extract_json_string(raw_text: str) -> str:
        """
        Attempts to extract a JSON string from a text output. 
        It strips markdown block tags and extracts the first matching outer bracketed object {}.

        Args:
            raw_text: The raw output text from the model.

        Returns:
            str: The extracted JSON string candidate.
        """
        if not raw_text:
            return ""

        cleaned = raw_text.strip()

        # Remove markdown code block surrounds if present (e.g. ```json ... ``` or ``` ...)
        # Regex matches block format and captures internal content
        markdown_pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match_markdown = re.match(markdown_pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match_markdown:
            cleaned = match_markdown.group(1).strip()

        # If it still isn't simple, find the first '{' and last '}'
        # This handles cases where there's leading/trailing explanatory text
        try:
            start_idx = cleaned.index("{")
            end_idx = cleaned.rindex("}") + 1
            cleaned = cleaned[start_idx:end_idx]
        except ValueError:
            # If no brackets exist, we return the string as is and let JSON parser fail gracefully
            pass

        return cleaned

    @classmethod
    def parse(cls, raw_text: str) -> Dict[str, Any]:
        """
        Extracts and parses JSON from raw LLM text.
        If parsing fails, returns a default schema with null values.

        Args:
            raw_text: Raw response string from the model.

        Returns:
            Dict[str, Any]: Parsed JSON data or fallback dictionary.
        """
        fallback = {
            "min_value": None,
            "max_value": None,
            "unit": None
        }

        if not raw_text:
            logger.warning("Empty response received. Returning fallback.")
            return fallback

        json_candidate = cls.extract_json_string(raw_text)
        if not json_candidate:
            logger.warning("Could not identify JSON structure in raw text: %s", raw_text)
            return fallback

        try:
            parsed_data = json.loads(json_candidate)
            if not isinstance(parsed_data, dict):
                logger.warning("Parsed JSON is not a dictionary: %s", parsed_data)
                return fallback
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(
                "JSON decode error. Candidate string was: '%s'. Error: %s", 
                json_candidate, 
                e
            )
            return fallback
