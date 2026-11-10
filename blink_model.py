"""Compatibility shim; model code lives in backend.blink_model."""
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("backend.blink_model", run_name="__main__")
else:
    from backend import blink_model as _implementation
    sys.modules[__name__] = _implementation
