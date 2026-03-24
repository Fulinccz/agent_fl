"""Agent provider abstractions.

This package provides a simple adapter layer for selecting between local and online
(or remote) model providers. It is intentionally lightweight as a starting point for
building a pluggable inference stack.
"""

from .registry import get_agent

__all__ = ["get_agent"]
