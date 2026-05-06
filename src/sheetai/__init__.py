"""sheetai - Process spreadsheet rows through LLM APIs."""

__version__ = "0.1.0"

from .processor import SheetProcessor
from .config import ProcessingConfig

__all__ = ["SheetProcessor", "ProcessingConfig"]
