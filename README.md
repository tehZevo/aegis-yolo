# Aegis YOLO (v3) node

Object detection as an Aegis node. Currently powered by YOLO v3.

## Usage
### Input
POST a JSON string:
```js
"<base64 encoded image>"
```
to `/` to get a list of boxes like so:
```js
[
  {
    bounds: [x, y, w, h],
    confidence: 0.5, //0..1 range
    class_name: "car", //coco dataset class
    class_id: 2, //coco dataset class id
  }
  ...
]
```
or to `/annotate` to get an annotated (boxes, labels, confidences) base64 encoded image.

## Environment
- `PORT` - the port to listen on (defaults to 80)
- `MIN_CONFIDENCE` - minimum confidence to accept bounding box (defaults to 0.5)

## TODO:
* support other (newer) yolo models
* perhaps combine some of the aegis cv nodes into a single node?
