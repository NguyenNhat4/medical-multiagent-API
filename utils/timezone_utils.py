"""
Timezone utilities for handling Vietnam timezone
"""

from datetime import datetime, timezone, timedelta
import logging


# Vietnam timezone (UTC+7)
VIETNAM_TZ = timezone(timedelta(hours=7))


def get_vietnam_time() -> datetime:
    """
    Get current time in Vietnam timezone (UTC+7)
    
    Returns:
        datetime: Current datetime in Vietnam timezone
    """
    return datetime.now(VIETNAM_TZ)


def get_vietnam_time_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current time in Vietnam timezone as formatted string
    
    Args:
        fmt: Format string for datetime (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        str: Formatted datetime string in Vietnam timezone
    """
    return get_vietnam_time().strftime(fmt)


def convert_utc_to_vietnam(utc_dt: datetime) -> datetime:
    """
    Convert UTC datetime to Vietnam timezone
    
    Args:
        utc_dt: UTC datetime (can be timezone naive or aware)
        
    Returns:
        datetime: Datetime converted to Vietnam timezone
    """
    # If timezone naive, assume it's UTC
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    
    return utc_dt.astimezone(VIETNAM_TZ)


class VietnamFormatter(logging.Formatter):
    """
    Custom logging formatter that displays timestamps in Vietnam timezone
    """
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        
    def formatTime(self, record, datefmt=None):
        """
        Override formatTime to use Vietnam timezone
        """
        # Convert record time to Vietnam timezone
        vietnam_time = datetime.fromtimestamp(record.created, VIETNAM_TZ)
        
        if datefmt:
            return vietnam_time.strftime(datefmt)
        else:
            return vietnam_time.strftime("%Y-%m-%d %H:%M:%S")


def setup_vietnam_logging(logger_name: str = None, 
                         level: int = logging.INFO,
                         format_str: str = "%(asctime)s [VN] - %(name)s - %(levelname)s - %(message)s") -> logging.Logger:
    """
    Setup logging with Vietnam timezone
    
    Args:
        logger_name: Name of the logger (if None, uses root logger)
        level: Logging level
        format_str: Format string for log messages
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False 
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with Vietnam timezone formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = VietnamFormatter(format_str)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# Example usage
if __name__ == "__main__":
    # Test timezone utilities
    print(f"Current Vietnam time: {get_vietnam_time()}")
    print(f"Vietnam time string: {get_vietnam_time_str()}")
    
    # Test logging
    test_logger = setup_vietnam_logging("test_logger")
    test_logger.info("This is a test log message with Vietnam timezone")
    
    # Test UTC conversion
    utc_now = datetime.utcnow()
    vietnam_time = convert_utc_to_vietnam(utc_now)
    print(f"UTC time: {utc_now}")
    print(f"Vietnam time: {vietnam_time}")
