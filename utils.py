import cv2
import base64
import numpy as np

def b64_to_img(b64):
  if b64.startswith("data:image/"):
    b64 = b64[b64.index(","):]
  nparr = np.fromstring(base64.b64decode(b64), np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img

def img_to_b64(img, format=".jpg"):
  retval, img = cv2.imencode(format, img)
  img = base64.b64encode(img)
  img = img.decode("ascii")
  return img
