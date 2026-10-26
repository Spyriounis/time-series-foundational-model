"""
A from-scratch foundation model for zero-shot probabilistic time-series forecasting.

The package is a scaffold: modelling, data and training modules land in later stages.
"""

from importlib.metadata import version

__version__: str = version("tsfm")

__all__ = ["__version__"]
