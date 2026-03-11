"""
LangSmith configuration - LangGraph handles tracing automatically
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Environment Configuration - LangGraph reads these automatically
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# No-op decorator - we don't need it anymore
def trace(name: str, tags: list = None):
    """No-op decorator - LangGraph handles tracing"""
    def decorator(func):
        return func
    return decorator

# Placeholder for compatibility
langsmith = None

def attach_metadata(metadata: dict):
    """No-op - LangGraph handles metadata"""
    pass

def get_current_run():
    """No-op"""
    return None

# Log config on import
if LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2.lower() == "true":
    logger.info(f"LangSmith tracing enabled for project: {LANGCHAIN_PROJECT}")
else:
    logger.warning("LangSmith tracing disabled - check your .env file")