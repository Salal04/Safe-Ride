"""Compatibility launcher for tools.deepsort."""
import runpy

if __name__ == "__main__":
    runpy.run_module("tools.deepsort", run_name="__main__")
else:
    from tools.deepsort import *  # noqa: F401,F403
