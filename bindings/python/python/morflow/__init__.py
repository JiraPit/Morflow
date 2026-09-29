"""
Morflow Python Interface
~~~~~~~~~~~~~~~~~~~~~~~~
High-performance modular dataflow pipeline engine for image, audio, and tensor computing.
"""

try:
    from ._morflow import load, from_str, Pipeline, run_cli
except ImportError as e:
    raise ImportError(
        f"Failed to import Morflow native extension module: {e}. "
        "Please build the extension using 'maturin develop' or 'pip install .'"
    ) from e

__all__ = ["load", "from_str", "Pipeline", "run_cli"]
__version__ = "0.1.2"
