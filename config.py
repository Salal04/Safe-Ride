"""Central Colab runtime, model paths, and SafeRide Vision settings.

Keep model files in WEIGHTS_DIR on the mounted Google Drive. The Colab
notebook validates and uploads the files listed in REQUIRED_WEIGHTS.
"""

from pathlib import Path


# Colab workspace and persistent Drive locations
COLAB_ROOT = Path("/content")
BASE_DIR = COLAB_ROOT / "deepSort"
DRIVE_DIR = COLAB_ROOT / "drive" / "MyDrive"
WEIGHTS_DIR = DRIVE_DIR / "Motor-Bikes-Data" / "local_project_weights"
VIDEO_PATH = BASE_DIR / "Input"
LEGACY_VIDEO_PATH = VIDEO_PATH / "vid1.mp4"
LEGACY_IMAGE_PATH = VIDEO_PATH / "img1.jpg"
LEGACY_TEST_IMAGE_DIR = BASE_DIR / "test" / "testimage"
LEGACY_OUTPUT_DIR = BASE_DIR / "mirror-indicator-yolov10m"
OUTPUT_PATH = DRIVE_DIR / "output-Bike"
LOG_DIR = OUTPUT_PATH / "Logs"
PUBLIC_OUTPUT_DIR = OUTPUT_PATH / "public"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_PUBLIC_DIR = BASE_DIR / "output_public"
RUN_LOG_DIR = BASE_DIR / "run_logs"
MANIFEST_PATH = BASE_DIR / "manifest.json"

# All trained/downloaded model weights live in Drive, not in the Git checkout.
YOLO_BIKE_MODEL_PATH = WEIGHTS_DIR / "yolov10m.pt"
MIRROR_MODEL_PATH = WEIGHTS_DIR / "bestM.pt"
INDICATOR_MODEL_PATH = WEIGHTS_DIR / "indicator_yolov12m_best.pt"
REID_WEIGHTS = WEIGHTS_DIR / "model.pth.tar-50"
OSNET_IMAGENET_WEIGHTS = WEIGHTS_DIR / "osnet_x1_0_imagenet.pth"
MARS_REID_WEIGHTS = WEIGHTS_DIR / "mars-small128.pb"
_ORIENTATION_NAMES = ("oreintation detector.pt", "orientation detector.pt")
ORIENTATION_MODEL_PATH = next(
    (WEIGHTS_DIR / name for name in _ORIENTATION_NAMES if (WEIGHTS_DIR / name).is_file()),
    WEIGHTS_DIR / _ORIENTATION_NAMES[0],
)
BLINK_WEIGHTS = WEIGHTS_DIR / "best_model.pth"
BIKE_PLATE_MODEL_PATH = WEIGHTS_DIR / "best (4).pt"
BIKE_MODEL_PATH = BIKE_PLATE_MODEL_PATH
PLATE_MODEL_PATH = BIKE_PLATE_MODEL_PATH
REQUIRED_WEIGHTS = {
    "bike_detector": YOLO_BIKE_MODEL_PATH,
    "mirror_detector": MIRROR_MODEL_PATH,
    "indicator_detector": INDICATOR_MODEL_PATH,
    "reid": REID_WEIGHTS,
    "orientation": ORIENTATION_MODEL_PATH,
    "blink": BLINK_WEIGHTS,
    "bike_plate_detector": BIKE_PLATE_MODEL_PATH,
}
OPTIONAL_WEIGHTS = {
    # The active pipeline uses the project-trained checkpoint above. This
    # ImageNet checkpoint is available for alternate OSNet experiments.
    "osnet_imagenet_pretraining": OSNET_IMAGENET_WEIGHTS,
    # Only the legacy deepsort.py script uses this TensorFlow encoder.
    "legacy_mars_reid": MARS_REID_WEIGHTS,
}

# Backend settings
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 5000
LOG_LEVEL = "DEBUG"

# Runtime / model settings
DEVICE = "auto"
INFERENCE_IMAGE_SIZE = 1280
LEGACY_IMAGE_SIZE = 640
LEGACY_DETECTOR_CONFIDENCE = 0.4
MAX_IOU_DISTANCE = 0.7
OSNET_MODEL_NAME = "osnet_x1_0"
OSNET_INPUT_SIZE = (256, 256)
BIKE_CLASS_ID = 3
GOT_OCR_MODEL_ID = "stepfun-ai/GOT-OCR2_0"
GOT_OCR_TYPE = "ocr"
GOT_OCR_TMP_PATH = "/tmp/_plate_detector_got_ocr_crop.jpg"
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
PLATE_CLASS_ID = None
PLATE_MAX_AREA_RATIO = 0.40
IMAGE_SAVE_DIR = str(BASE_DIR / "received_images")

# Detection, tracking, and saved-image thresholds
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 1.0
N_INIT = 3
MAX_AGE = 30
DRAW_TRAILS = True
TRAIL_LEN = 20
MOTORCYCLE_CLASS_ID = 3
MIN_BIKE_AREA_FOR_BEST_IMAGE = 6000
PLATE_CONF = 0.3
BIKE_CONF = 0.3
OCR_CONF = 0.35
PLATE_VIDEO_PROGRESS_EVERY = 10
PLATE_VIDEO_DEBUG_EVERY = 1
MIN_PLATE_AREA_FOR_SAVE = 400
MIN_SLOT_HITS_FOR_BOTH = 3
TURN_ZONES = []
TURN_ZONE_OVERLAP_THRESHOLD = 0.30
TURN_ZONE_DRAW_COLOR = (255, 255, 0)
TURN_ZONE_DRAW_THICKNESS = 2

# Blink classifier architecture
BLINK_MODEL_CONFIG = {
    "cnn_name": "resnet18",
    "seq_type": "RNN",
    "num_layers": 2,
    "hidden_size": 256,
    "dropout": 0.5,
}

# Per-frame profiler settings
PROFILE_ENABLED = True
PROFILE_SYNC_CUDA = True
PROFILE_WARMUP_FRAMES = 3
PROFILE_LOG_EVERY_FRAME = True
PROFILE_SUMMARY_EVERY = 50
PROFILE_SLOW_FRAME_FACTOR = 2.0
PROFILE_GPU_SAMPLE_INTERVAL = 0.1
PROFILE_SAVE_DIR_NAME = "profile"
GPU_SPECS = {
    "A10G": (31.2, 600, 24),
    "A100": (19.5, 1555, 40),
    "H100": (67.0, 3350, 80),
    "V100": (15.7, 900, 16),
    "RTX 4090": (82.6, 1008, 24),
    "RTX 3090": (35.6, 936, 24),
    "L4": (30.3, 300, 24),
    "T4": (8.1, 320, 16),
}
GPU_STAGE_HINTS = (
    ".model_call",
    "reid.extract",
    "blink.forward",
    "plate.detect",
    "ocr.read",
)

# Compatibility names used by the older standalone deepsort.py script.
YOLO_MODEL = str(YOLO_BIKE_MODEL_PATH)
Mirror_indicater_DETECTOR = str(MIRROR_MODEL_PATH)
