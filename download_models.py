import urllib.request
from os.path import exists
import os

def download_models():
  os.makedirs("models", exist_ok=True)

  YOLO_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg"
  YOLO_CFG_PATH = "models/yolov3.cfg"
  if not exists(YOLO_CFG_PATH):
    print(f"downloading {YOLO_CFG_URL} to {YOLO_CFG_PATH}")
    urllib.request.urlretrieve(YOLO_CFG_URL, YOLO_CFG_PATH)

  YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
  YOLO_WEIGHTS_PATH = "models/yolov3.weights"
  if not exists(YOLO_WEIGHTS_PATH):
    print(f"downloading {YOLO_WEIGHTS_URL} to {YOLO_WEIGHTS_PATH}")
    urllib.request.urlretrieve(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)

if __name__ == "__main__":
  download_models()
