"""Time utilities."""

from datetime import datetime

from adaptive_oscillator.definitions import TIME_FORMAT


def time_str_to_seconds(time_str: str) -> float:
    """Convert a time string to seconds."""
    dt = datetime.strptime(time_str, TIME_FORMAT)
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
