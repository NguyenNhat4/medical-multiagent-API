"""
Response parser utility for cleaning and structuring medical responses
Enhanced with safety features and robust error handling
"""

import yaml
import json
import re
import logging
import textwrap
from typing import Any, Dict, Optional, List, Union, Tuple
from functools import wraps
import time

# Configure logging with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))

# Safety constants
MAX_RESPONSE_SIZE = 50000  # 50KB max response size
PARSING_TIMEOUT = 3  # 10 seconds max parsing time


def timeout_protection(timeout_seconds: int):
    """Decorator to add timeout protection to parsing functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(f"Function {func.__name__} took {elapsed:.2f}s (exceeded {timeout_seconds}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {elapsed:.2f}s: {e}")
                return None
        return wrapper
    return decorator


def safe_size_check(response: str) -> bool:
    """Check if response size is within safe limits"""
    if not response:
        return False
    
    size = len(response.encode('utf-8'))
    if size > MAX_RESPONSE_SIZE:
        logger.warning(f"Response size {size} bytes exceeds limit {MAX_RESPONSE_SIZE}")
        return False
    
    return True


@timeout_protection(PARSING_TIMEOUT)
def parse_yaml_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Robust YAML parsing utility with multiple fallback strategies.
    Enhanced with safety checks and timeout protection.
    
    Strategies in order:
    1. Safety checks (size, type validation)
    2. Flexible code fence detection (```yaml, ```YAML, ```yml, etc.)
    3. Try parsing the entire response as YAML
    4. Use regex to extract YAML-like content
    5. Fall back to JSON parsing
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed dictionary or None if all strategies fail
    """
    if not response or not isinstance(response, str):
        logger.warning("parse_yaml_response: Invalid input")
        return None
    
    response = response.strip()
    
    # Safety check: response size
    if not safe_size_check(response):
        logger.warning("parse_yaml_response: Response failed safety size check")
        return None
    
    # Strategy 1: Flexible code fence detection
    yaml_content = _extract_from_code_fences(response)
    if yaml_content:
        try:
            cleaned = textwrap.dedent(yaml_content).strip()
            result = yaml.safe_load(cleaned)
            if isinstance(result, dict):
                logger.info("parse_yaml_response: Success with code fence extraction")
                return result
        except Exception as e:
            logger.debug(f"parse_yaml_response: Code fence extraction failed: {e}")
    
    # Strategy 2: Try parsing entire response as YAML
    try:
        result = yaml.safe_load(response)
        if isinstance(result, dict):
            logger.info("parse_yaml_response: Success with full response parsing")
            return result
    except Exception as e:
        logger.debug(f"parse_yaml_response: Full response parsing failed: {e}")
    
    # Strategy 3: Use regex to extract YAML-like content
    yaml_content = _extract_with_regex(response)
    if yaml_content:
        try:
            result = yaml.safe_load(yaml_content)
            if isinstance(result, dict):
                logger.info("parse_yaml_response: Success with regex extraction")
                return result
        except Exception as e:
            logger.debug(f"parse_yaml_response: Regex extraction failed: {e}")
    
    # Strategy 4: Fall back to JSON parsing
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            logger.info("parse_yaml_response: Success with JSON fallback")
            return result
    except Exception as e:
        logger.debug(f"parse_yaml_response: JSON fallback failed: {e}")
    
    logger.warning("parse_yaml_response: All parsing strategies failed")
    return None


def _extract_from_code_fences(response: str) -> Optional[str]:
    """
    Extract YAML content from various code fence patterns.
    
    Supports:
    - ```yaml ... ```
    - ```YAML ... ```
    - ```yml ... ```
    - ```YML ... ```
    - ``` ... ``` (generic)
    """
    # A more robust pattern to capture content within fences
    pattern = r'```(yaml|yml)?\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(2).strip()
        if content:
            logger.debug(f"_extract_from_code_fences: Found content with robust pattern")
            return _clean_yaml_content(content)

    # Fallback to original patterns if the above fails
    yaml_patterns = [
        r'```yaml\s*\n(.*?)\n\s*```',
        r'```YAML\s*\n(.*?)\n\s*```', 
        r'```yml\s*\n(.*?)\n\s*```',
        r'```YML\s*\n(.*?)\n\s*```',
        r'```\s*\n(.*?)\n\s*```',  # Generic code fence
        # Alternative patterns for edge cases
        r'```yaml(.*?)```',
        r'```YAML(.*?)```',
        r'```yml(.*?)```',
        r'```(.*?)```',  # Most generic
    ]
    
    for pattern in yaml_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content:
                logger.debug(f"_extract_from_code_fences: Found content with pattern {pattern}")
                return _clean_yaml_content(content)
    
    return None


def _clean_yaml_content(content: str) -> str:
    """
    Clean YAML content to fix common parsing issues.
    
    Specifically handles:
    - URLs in sources lists that need proper quoting
    - Basic structure cleanup
    """
    if not content:
        return content
    
    lines = content.split('\n')
    cleaned_lines = []
    in_sources_section = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detect sources section
        if stripped.startswith('sources:'):
            in_sources_section = True
            cleaned_lines.append(line)
            continue
        
        # Detect end of sources section (next top-level key)
        if in_sources_section and stripped and not stripped.startswith('-') and not stripped.startswith(' ') and ':' in stripped:
            in_sources_section = False
        
        # Process lines in sources section
        if in_sources_section and stripped.startswith('-'):
            # Check if source line needs quoting
            source_content = stripped[1:].strip()  # Remove the dash
            if source_content and not (source_content.startswith('"') and source_content.endswith('"')):
                # Add quotes if not already quoted
                indent = len(line) - len(line.lstrip())
                cleaned_lines.append(' ' * indent + f'- "{source_content}"')
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def _extract_with_regex(response: str) -> Optional[str]:
    """
    Use regex patterns to extract YAML-like content from the response.
    
    Looks for:
    - Key-value patterns (key: value)
    - List patterns (- item)
    - Nested structures
    """
    yaml_blocks = []
    
    # Pattern 1: Look for key-value pairs followed by potential YAML content
    key_value_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s*:\s*(?:[^\n]*\n)*)'
    matches = re.finditer(key_value_pattern, response)
    
    for match in matches:
        start = match.start()
        end = match.end()
        
        remaining = response[end:]
        lines = remaining.split('\n')
        
        yaml_lines = []
        for line in lines[:20]:  # Limit to 20 lines
            line = line.strip()
            if not line:
                continue
            # Check if line looks like YAML (key: value, - item, or indented)
            if (re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*:', line) or  # key: value
                re.match(r'^\s*-\s+', line) or  # - item
                re.match(r'^\s+', line)):       # indented
                yaml_lines.append(line)
            else:
                break
            
        if yaml_lines:
            block = match.group(1) + '\n'.join(yaml_lines)
            yaml_blocks.append(block)
    
    if yaml_blocks:
        largest_block = max(yaml_blocks, key=len)
        logger.debug(f"_extract_with_regex: Found YAML-like block of {len(largest_block)} chars")
        return largest_block
    
    return None



def validate_yaml_structure(data: Dict[str, Any], required_fields: list = None, 
                           optional_fields: list = None, field_types: Dict[str, type] = None,
                           allow_extra_fields: bool = True) -> bool:
    """
    Enhanced validation that parsed YAML has the expected structure.
    
    Args:
        data: Parsed dictionary
        required_fields: List of required field names
        optional_fields: List of optional field names
        field_types: Dict mapping field names to expected types
        allow_extra_fields: Whether to allow fields not in required/optional lists
        
    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(data, dict):
        logger.warning("validate_yaml_structure: Data is not a dictionary")
        return False
    
    if not data:
        logger.warning("validate_yaml_structure: Empty dictionary")
        return False
    
    # Check required fields
    if required_fields:
        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"validate_yaml_structure: Missing required fields: {missing_fields}")
            return False
    
    # Check field types
    if field_types:
        type_errors = []
        for field, expected_type in field_types.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    type_errors.append(f"'{field}': expected {expected_type.__name__}, got {type(data[field]).__name__}")
        
        if type_errors:
            logger.warning(f"validate_yaml_structure: Type validation errors: {type_errors}")
            return False
    
    # Check unexpected fields (only if strict mode)
    if not allow_extra_fields and required_fields and optional_fields:
        all_valid_fields = set(required_fields) | set(optional_fields)
        unexpected_fields = set(data.keys()) - all_valid_fields
        if unexpected_fields:
            logger.warning(f"validate_yaml_structure: Unexpected fields: {list(unexpected_fields)}")
            return False
    
    return True


def parse_yaml_with_schema(response: str, required_fields: List[str] = None,
                          optional_fields: List[str] = None,
                          field_types: Dict[str, type] = None) -> Optional[Dict[str, Any]]:
    """
    Parse YAML response and validate against schema in one step.
    
    Args:
        response: Raw LLM response string
        required_fields: List of required field names
        optional_fields: List of optional field names  
        field_types: Dict mapping field names to expected types
        
    Returns:
        Validated parsed dictionary or None if parsing/validation fails
    """
    # Parse the response
    parsed_data = parse_yaml_response(response)
    
    if parsed_data is None:
        logger.warning("parse_yaml_with_schema: Failed to parse response")
        return None
    
    # Validate the structure
    is_valid = validate_yaml_structure(
        parsed_data, 
        required_fields=required_fields,
        optional_fields=optional_fields,
        field_types=field_types
    )
    
    if not is_valid:
        logger.warning("parse_yaml_with_schema: Response failed schema validation")
        return None
    
    logger.info("parse_yaml_with_schema: Successfully parsed and validated response")
    return parsed_data


# --- Simple high-level helpers expected by API layer ---
def _fallback_summary(text: str, max_len: int = 200) -> str:
    try:
        # Take the first sentence or trim to max_len
        sentence_end = max(text.find("."), text.find("!"), text.find("?"))
        if sentence_end != -1:
            candidate = text[: sentence_end + 1].strip()
            if candidate:
                return candidate[:max_len]
        return text[:max_len].rstrip()
    except Exception:
        return text[:max_len].rstrip()


def _normalize_suggestions(suggestions: Optional[List[str]], fallback: List[str]) -> List[str]:
    if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
        return suggestions[:5]
    return fallback


def parse_medical_response(raw_response: str,
                           raw_suggestions: Optional[List[str]] = None
                           ) -> Tuple[str, str, List[str]]:
    """
    Parse a potentially structured medical response. Prefer YAML/JSON structure with keys:
    - explanation (str)
    - summary (str)
    - questionSuggestion (list[str])

    Returns a tuple: (explanation, summary, questionSuggestion)
    """
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)

    parsed = parse_yaml_response(raw_response)

    if isinstance(parsed, dict):
        # Support multiple key variants from different prompt versions
        explanation = (
          
             parsed.get("explanation")
            or parsed.get("answer")
            or raw_response
        )
        summary = (
            parsed.get("sumary")  # new misspelled key per prompt schema
            or parsed.get("summary")
            or _fallback_summary(str(explanation))
        )
        suggestions = parsed.get("questionSuggestion") or parsed.get("questions") or raw_suggestions
        suggestions = _normalize_suggestions(
            suggestions,
            [
                "Bạn có triệu chứng nào khác không?",
                "Tình trạng này kéo dài bao lâu rồi?",
                "Bạn có bệnh nền hay đang dùng thuốc gì không?"
            ]
        )
        return str(explanation), str(summary), suggestions

    # Fallback: treat entire response as explanation
    explanation = raw_response.strip()
    summary = _fallback_summary(explanation)
    suggestions = _normalize_suggestions(
        raw_suggestions,
        [
            "Bạn muốn làm rõ điểm nào thêm?",
            "Bạn có thể mô tả triệu chứng chi tiết hơn không?",
            "Bạn có xét nghiệm hay dữ liệu liên quan không?"
        ]
    )
    return explanation, summary, suggestions


def handle_greeting_response(raw_response: str) -> Tuple[str, str, List[str]]:
    """
    Specialized formatting for greeting inputs.
    """
    explanation = (raw_response or "Xin chào! Tôi có thể hỗ trợ gì cho bạn hôm nay?").strip()
    summary = ""
    suggestions = [
        "Tôi muốn hỏi về một vấn đề răng miệng.",
        "Tôi có triệu chứng X, cần tư vấn.",
        "Tôi muốn biết cách chăm sóc răng miệng."
    ]
    return explanation, summary, suggestions


def handle_statement_response(raw_response: str,
                              raw_suggestions: Optional[List[str]] = None
                              ) -> Tuple[str, str, List[str]]:
    """
    Specialized formatting for statement inputs.
    """
    explanation = (raw_response or "Tôi đã ghi nhận thông tin của bạn.").strip()
    summary = _fallback_summary(explanation)
    suggestions = _normalize_suggestions(
        raw_suggestions,
        [
            "Bạn có câu hỏi cụ thể nào về tình trạng này?",
            "Bạn mong muốn hướng xử trí/điều trị như thế nào?",
            "Bạn có thông tin bổ sung (thời gian, mức độ, yếu tố nặng/nhẹ)?"
        ]
    )
    return explanation, summary, suggestions
