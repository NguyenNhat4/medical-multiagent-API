# Core framework import
from pocketflow import Node

# Standard library imports
import logging

# Configure logging for this module with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))



class FallbackNode(Node):
    """Node fallback khi API quá tải - retrieve query và trả kết quả dựa trên score"""
    
    def prep(self, shared):
        return {}
        

    def exec(self, inputs):
        return {}
    
    def post(self, shared, prep_res, exec_res):
        logger.info("🔄 [FallbackNode] POST - Lưu fallback response")
        shared["explain"] = "Xin lỗi bạn hãy thử lại sau vài phút nữa, hiện tại hệ thống đang quá tải."
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        return "default"

