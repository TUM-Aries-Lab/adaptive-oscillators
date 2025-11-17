"""Adaptive Oscillator."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib

try:
    __version__ = version("adaptive-oscillator")
except PackageNotFoundError:
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        __version__ = dict(tomllib.load(f))["tool"]["poetry"]["version"]

__author__ = "Tony Smoragiewicz"
__email__ = "tony.smoragiewicz@tum.de"
