"""Stable root launcher for the Flask backend."""
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("backend.api_server", run_name="__main__")
else:
    from backend import api_server as _implementation
    sys.modules[__name__] = _implementation
