import os

import cv2 as cv
import numpy as np
from protopost import ProtoPost

from download_models import download_models
from utils import b64_to_img, img_to_b64

download_models()

PORT = int(os.getenv("PORT", 80))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.5))

class_names = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('models/yolov3.cfg', 'models/yolov3.weights')
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# determine the output layer
ln = net.getLayerNames()
unconnected_outs = np.array(net.getUnconnectedOutLayers()).flatten().tolist()
print(unconnected_outs)
ln = [ln[i - 1] for i in unconnected_outs]

def get_boxes(img):
  blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
  net.setInput(blob)
  outputs = net.forward(ln)

  boxes = []
  confidences = []
  class_ids = []
  h, w = img.shape[:2]

  ret_boxes = []

  for output in outputs:
    for detection in output:
      scores = detection[5:]
      class_id = int(np.argmax(scores))
      confidence = scores[class_id]
      if confidence > MIN_CONFIDENCE:
        box = detection[:4] * np.array([w, h, w, h]) #TODO: maybe dont do this?
        (centerX, centerY, width, height) = box.astype("int")
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        box = [x, y, int(width), int(height)]
        box = np.array(box).tolist() #just in case
        boxes.append(box)
        confidences.append(float(confidence))
        class_ids.append(class_id)

  indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  if len(indices) > 0:
    for i in indices.flatten():
      x, y, w, h = boxes[i][0:4]
      ret_boxes.append({
        "bounds": [x, y, w, h],
        "confidence": confidences[i],
        "class": class_names[class_ids[i]],
        "class_id": class_ids[i]
      })

  return ret_boxes

def protopost_get_boxes(data):
  #load image from base 64
  img = b64_to_img(data)
  boxes = get_boxes(img)
  return boxes

def protopost_annotate(data):
  #load image from base 64
  img = b64_to_img(data)
  #get bounding boxes, classes, and confidences
  boxes = get_boxes(img)

  for box in boxes:
    x, y, w, h = box["bounds"]
    confidence = box["confidence"]
    class_name = box["class"]
    class_id = box["class_id"]

    #draw rectangle
    color = [int(c) for c in colors[class_id]]
    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(class_name, confidence)
    cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

  #conver image back to base 64
  img = img_to_b64(img)

  return img

routes = {
  "": protopost_get_boxes,
  "annotate": protopost_annotate
}

ProtoPost(routes).start(PORT)
