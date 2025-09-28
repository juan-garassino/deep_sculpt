"""
Modern logger with Rich integration for DeepSculpt v2.0.

This module provides enhanced logging capabilities with Rich library integration
for beautiful console output, progress tracking, and structured logging.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TextIO
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import threading
from enum import Enum

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn,
        TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.live import Live
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console class
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        
        def log(self, *args, **kwargs):
            print(*args)


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"
    PROGRESS = "PROGRESS"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    context: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


class RichLogger:
    """
    Modern logger with Rich integration for enhanced console output.
    
    Features:
    - Beautiful console output with colors and formatting
    - Progress bars and spinners
    - Structured logging with context
    - File and console output
    - Real-time training progress visualization
    - Hierarchical logging with sections
    """
    
    def __init__(
        self,
        name: str = "DeepSculpt",
        level: Union[str, LogLevel] = LogLevel.INFO,
        console_output: bool = True,
        file_output: Optional[str] = None,
        rich_tracebacks: bool = True,
        show_time: bool = True,
        show_path: bool = False,
        markup: bool = True,
        width: Optional[int] = None
    ):
        """
        Initialize Rich logger.
        
        Args:
            name: Logger name
            level: Logging level
            console_output: Whether to output to console
            file_output: File path for log output
            rich_tracebacks: Whether to use Rich tracebacks
            show_time: Whether to show timestamps
            show_path: Whether to show file paths
            markup: Whether to enable Rich markup
            width: Console width (auto-detect if None)
        """
        self.name = name
        self.level = level if isinstance(level, LogLevel) else LogLevel(level)
        self.console_output = console_output
        self.file_output = file_output
        self.markup = markup
        
        # Initialize Rich console
        if RICH_AVAILABLE:
            self.console = Console(
                width=width,
                force_terminal=True,
                markup=markup,
                highlight=True
            )
        else:
            self.console = Console()
        
        # Initialize Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add Rich handler for console output
        if console_output and RICH_AVAILABLE:
            rich_handler = RichHandler(
                console=self.console,
                show_time=show_time,
                show_path=show_path,
                rich_tracebacks=rich_tracebacks,
                markup=markup
            )
            rich_handler.setLevel(getattr(logging, self.level.value))
            self.logger.addHandler(rich_handler)
        elif console_output:
            # Fallback to standard handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if file_output:
            self._setup_file_handler(file_output)
        
        # Progress tracking
        self.progress_bars: Dict[str, TaskID] = {}
        self.current_progress: Optional[Progress] = None
        self.progress_lock = threading.Lock()
        
        # Section tracking
        self.section_stack: List[str] = []
        self.section_start_times: Dict[str, float] = {}
        
        # Log entries for structured logging
        self.log_entries: List[LogEntry] = []
        self.max_entries = 1000  # Keep last 1000 entries
        
        # Training metrics tracking
        self.training_metrics: Dict[str, List[float]] = {}
        self.current_epoch = 0
        self.current_step = 0
    
    def _setup_file_handler(self, file_path: str):
        """Setup file handler for logging."""
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(file_path)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, self.level.value))
        self.logger.addHandler(file_handler)
    
    def _add_log_entry(self, level: LogLevel, message: str, **kwargs):
        """Add structured log entry."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            message=message,
            context=kwargs.get('context'),
            module=kwargs.get('module'),
            function=kwargs.get('function'),
            line=kwargs.get('line'),
            extra=kwargs.get('extra')
        )
        
        self.log_entries.append(entry)
        
        # Keep only recent entries
        if len(self.log_entries) > self.max_entries:
            self.log_entries = self.log_entries[-self.max_entries:]
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._add_log_entry(LogLevel.DEBUG, message, **kwargs)
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._add_log_entry(LogLevel.INFO, message, **kwargs)
        if RICH_AVAILABLE:
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._add_log_entry(LogLevel.WARNING, message, **kwargs)
        if RICH_AVAILABLE:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        else:
            self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._add_log_entry(LogLevel.ERROR, message, **kwargs)
        if RICH_AVAILABLE:
            self.console.print(f"[red]✗[/red] {message}")
        else:
            self.logger.error(message)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._add_log_entry(LogLevel.CRITICAL, message, **kwargs)
        if RICH_AVAILABLE:
            self.console.print(f"[red bold]💥[/red bold] {message}")
        else:
            self.logger.critical(message)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        self._add_log_entry(LogLevel.SUCCESS, message, **kwargs)
        if RICH_AVAILABLE:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            self.logger.info(f"SUCCESS: {message}")
    
    def print_panel(self, message: str, title: Optional[str] = None, style: str = "blue"):
        """Print message in a panel."""
        if RICH_AVAILABLE:
            panel = Panel(message, title=title, border_style=style)
            self.console.print(panel)
        else:
            if title:
                print(f"=== {title} ===")
            print(message)
            if title:
                print("=" * (len(title) + 8))
    
    def print_table(self, data: List[Dict[str, Any]], title: Optional[str] = None):
        """Print data as a formatted table."""
        if not data:
            return
        
        if RICH_AVAILABLE:
            table = Table(title=title, box=box.ROUNDED)
            
            # Add columns
            for key in data[0].keys():
                table.add_column(str(key).title(), style="cyan")
            
            # Add rows
            for row in data:
                table.add_row(*[str(value) for value in row.values()])
            
            self.console.print(table)
        else:
            if title:
                print(f"\n{title}")
                print("-" * len(title))
            
            # Simple table format
            if data:
                headers = list(data[0].keys())
                print(" | ".join(headers))
                print("-" * (len(" | ".join(headers))))
                for row in data:
                    print(" | ".join(str(row[key]) for key in headers))
    
    def print_syntax(self, code: str, language: str = "python", theme: str = "monokai"):
        """Print syntax-highlighted code."""
        if RICH_AVAILABLE:
            syntax = Syntax(code, language, theme=theme, line_numbers=True)
            self.console.print(syntax)
        else:
            print(f"```{language}")
            print(code)
            print("```")
    
    def print_tree(self, data: Dict[str, Any], title: str = "Data Structure"):
        """Print hierarchical data as a tree."""
        if RICH_AVAILABLE:
            tree = Tree(title)
            self._build_tree(tree, data)
            self.console.print(tree)
        else:
            print(f"\n{title}:")
            self._print_dict_tree(data)
    
    def _build_tree(self, tree, data, max_depth: int = 5, current_depth: int = 0):
        """Build Rich tree from data."""
        if current_depth >= max_depth:
            tree.add("[dim]...[/dim]")
            return
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)) and value:
                    branch = tree.add(f"[bold]{key}[/bold]")
                    self._build_tree(branch, value, max_depth, current_depth + 1)
                else:
                    tree.add(f"{key}: [green]{value}[/green]")
        elif isinstance(data, list):
            for i, item in enumerate(data[:10]):  # Limit to first 10 items
                if isinstance(item, (dict, list)) and item:
                    branch = tree.add(f"[bold][{i}][/bold]")
                    self._build_tree(branch, item, max_depth, current_depth + 1)
                else:
                    tree.add(f"[{i}]: [green]{item}[/green]")
            if len(data) > 10:
                tree.add(f"[dim]... and {len(data) - 10} more items[/dim]")
    
    def _print_dict_tree(self, data, indent: int = 0):
        """Print dictionary as tree (fallback)."""
        prefix = "  " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{prefix}{key}:")
                    self._print_dict_tree(value, indent + 1)
                else:
                    print(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                print(f"{prefix}[{i}]: {item}")
    
    @contextmanager
    def section(self, title: str, style: str = "bold blue"):
        """Context manager for logging sections."""
        self.begin_section(title, style)
        try:
            yield
        finally:
            self.end_section(title)
    
    def begin_section(self, title: str, style: str = "bold blue"):
        """Begin a logging section."""
        self.section_stack.append(title)
        self.section_start_times[title] = time.time()
        
        if RICH_AVAILABLE:
            rule = Rule(f"[{style}]{title}[/{style}]", style=style)
            self.console.print(rule)
        else:
            print(f"\n{'='*60}")
            print(f" {title}")
            print(f"{'='*60}")
    
    def end_section(self, title: Optional[str] = None):
        """End a logging section."""
        if not self.section_stack:
            return
        
        section_title = title or self.section_stack[-1]
        if section_title in self.section_stack:
            self.section_stack.remove(section_title)
        
        if section_title in self.section_start_times:
            duration = time.time() - self.section_start_times[section_title]
            del self.section_start_times[section_title]
            
            if RICH_AVAILABLE:
                self.console.print(f"[dim]Completed {section_title} in {duration:.2f}s[/dim]")
            else:
                print(f"Completed {section_title} in {duration:.2f}s")
    
    def create_progress_bar(
        self,
        name: str,
        total: int,
        description: str = "Processing",
        show_speed: bool = True
    ) -> str:
        """Create a progress bar."""
        with self.progress_lock:
            if not RICH_AVAILABLE:
                return name
            
            if self.current_progress is None:
                columns = [
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                ]
                
                if show_speed:
                    columns.append(TextColumn("• [progress.data_speed]{task.speed} it/s"))
                
                self.current_progress = Progress(*columns, console=self.console)
                self.current_progress.start()
            
            task_id = self.current_progress.add_task(description, total=total)
            self.progress_bars[name] = task_id
            
            return name
    
    def update_progress(self, name: str, advance: int = 1, description: Optional[str] = None):
        """Update progress bar."""
        with self.progress_lock:
            if name in self.progress_bars and self.current_progress:
                task_id = self.progress_bars[name]
                self.current_progress.update(task_id, advance=advance, description=description)
    
    def finish_progress(self, name: str):
        """Finish and remove progress bar."""
        with self.progress_lock:
            if name in self.progress_bars and self.current_progress:
                task_id = self.progress_bars[name]
                self.current_progress.remove_task(task_id)
                del self.progress_bars[name]
                
                if not self.progress_bars:
                    self.current_progress.stop()
                    self.current_progress = None
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        prefix: str = "train"
    ):
        """Log training step with metrics."""
        self.current_epoch = epoch
        self.current_step = step
        
        # Store metrics
        for key, value in metrics.items():
            metric_key = f"{prefix}_{key}"
            if metric_key not in self.training_metrics:
                self.training_metrics[metric_key] = []
            self.training_metrics[metric_key].append(value)
        
        # Format metrics for display
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        metric_text = " | ".join(metric_strs)
        
        if RICH_AVAILABLE:
            self.console.print(
                f"[bold]Epoch {epoch}[/bold] | "
                f"[bold]Step {step}[/bold] | "
                f"[cyan]{metric_text}[/cyan]"
            )
        else:
            print(f"Epoch {epoch} | Step {step} | {metric_text}")
    
    def log_training_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        epoch_time: Optional[float] = None
    ):
        """Log training epoch summary."""
        # Create summary table
        summary_data = []
        
        for key, value in train_metrics.items():
            row = {"Metric": f"Train {key}", "Value": f"{value:.6f}"}
            summary_data.append(row)
        
        if val_metrics:
            for key, value in val_metrics.items():
                row = {"Metric": f"Val {key}", "Value": f"{value:.6f}"}
                summary_data.append(row)
        
        if epoch_time:
            summary_data.append({"Metric": "Epoch Time", "Value": f"{epoch_time:.2f}s"})
        
        self.print_table(summary_data, title=f"Epoch {epoch} Summary")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.print_panel(
            f"Model: {model_info.get('name', 'Unknown')}\n"
            f"Parameters: {model_info.get('total_params', 0):,}\n"
            f"Trainable: {model_info.get('trainable_params', 0):,}\n"
            f"Device: {model_info.get('device', 'Unknown')}",
            title="Model Information",
            style="green"
        )
    
    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.print_tree(config, "Experiment Configuration")
    
    def export_logs(self, filepath: str, format: str = "json"):
        """Export logs to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump([asdict(entry) for entry in self.log_entries], f, indent=2)
        elif format == "csv":
            import csv
            with open(filepath, 'w', newline='') as f:
                if self.log_entries:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.log_entries[0]).keys())
                    writer.writeheader()
                    for entry in self.log_entries:
                        writer.writerow(asdict(entry))
        
        self.success(f"Logs exported to {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        summary = {
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "total_log_entries": len(self.log_entries),
            "metrics_tracked": list(self.training_metrics.keys()),
        }
        
        # Add latest metrics
        for key, values in self.training_metrics.items():
            if values:
                summary[f"latest_{key}"] = values[-1]
                summary[f"best_{key}"] = min(values) if "loss" in key else max(values)
        
        return summary
    
    def close(self):
        """Close logger and cleanup resources."""
        with self.progress_lock:
            if self.current_progress:
                self.current_progress.stop()
                self.current_progress = None
        
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
        
        self.logger.handlers.clear()


# Global logger instance
_global_logger: Optional[RichLogger] = None


def get_logger(
    name: str = "DeepSculpt",
    level: Union[str, LogLevel] = LogLevel.INFO,
    **kwargs
) -> RichLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = RichLogger(name=name, level=level, **kwargs)
    
    return _global_logger


def setup_logger(
    name: str = "DeepSculpt",
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[str] = None,
    **kwargs
) -> RichLogger:
    """Setup and configure global logger."""
    global _global_logger
    
    _global_logger = RichLogger(
        name=name,
        level=level,
        file_output=log_file,
        **kwargs
    )
    
    return _global_logger


# Convenience functions for backward compatibility
def begin_section(title: str, style: str = "bold blue"):
    """Begin a logging section."""
    logger = get_logger()
    logger.begin_section(title, style)


def end_section(title: Optional[str] = None):
    """End a logging section."""
    logger = get_logger()
    logger.end_section(title)


def log_action(message: str):
    """Log an action."""
    logger = get_logger()
    logger.info(message)


def log_success(message: str):
    """Log a success message."""
    logger = get_logger()
    logger.success(message)


def log_error(message: str):
    """Log an error message."""
    logger = get_logger()
    logger.error(message)


def log_warning(message: str):
    """Log a warning message."""
    logger = get_logger()
    logger.warning(message)


def log_info(message: str):
    """Log an info message."""
    logger = get_logger()
    logger.info(message)


# Context manager for sections (backward compatibility)
@contextmanager
def section(title: str, style: str = "bold blue"):
    """Context manager for logging sections."""
    logger = get_logger()
    with logger.section(title, style):
        yield