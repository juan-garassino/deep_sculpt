"""
Logger module for data transforms.

This module provides logging functions for the data transforms package,
importing from the main DeepSculpt logger utilities.
"""

# Import logging functions from the main utils logger
from ...utils.logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
    set_verbose,
    get_verbose,
    get_logger,
    setup_logger
)

# Re-export all functions for convenience
__all__ = [
    'begin_section',
    'end_section', 
    'log_action',
    'log_success',
    'log_error',
    'log_info',
    'log_warning',
    'set_verbose',
    'get_verbose',
    'get_logger',
    'setup_logger'
]