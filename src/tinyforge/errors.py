"""Errors a user can fix: shown as a plain message (exit code 2), never as a Python traceback."""

from __future__ import annotations


class UserError(RuntimeError):
    """Something the user must change (wrong model type, missing file...). Subclasses RuntimeError so
    the worker reports it as a failed job with the same message."""
