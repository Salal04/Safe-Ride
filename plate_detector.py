"""Compatibility shim; implementation lives in backend.plate_detector."""
import sys
from backend import plate_detector as _implementation
sys.modules[__name__] = _implementation
