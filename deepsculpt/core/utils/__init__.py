"""
Utilities package for DeepSculpt v2.0 PyTorch implementation.

This package contains utility functions and classes including
the modern Rich-integrated logger and other helper utilities.
"""

from .logger import (
    RichLogger,
    LogLevel,
    LogEntry,
    get_logger,
    setup_logger,
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_warning,
    log_info,
    section
)

__all__ = [
    "RichLogger",
    "LogLevel", 
    "LogEntry",
    "get_logger",
    "setup_logger",
    "begin_section",
    "end_section",
    "log_action",
    "log_success",
    "log_error",
    "log_warning",
    "log_info",
    "section",
]