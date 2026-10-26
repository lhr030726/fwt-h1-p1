"""Recorded performance control; leaves the scientific sampling recipe intact."""
import os


def validation_parallel_samples():
    """Read once per train invocation and include the result in provenance."""
    raw = os.environ.get('FWT_P1_VALIDATION_PARALLEL_SAMPLES', '1')
    if not raw.isascii() or not raw.isdecimal() or int(raw) < 1:
        raise ValueError('FWT_P1_VALIDATION_PARALLEL_SAMPLES must be a positive integer')
    return int(raw)
