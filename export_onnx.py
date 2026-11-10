"""Export configured YOLO weights to ONNX."""
import runpy
runpy.run_module("tools.export_onnx", run_name="__main__")
