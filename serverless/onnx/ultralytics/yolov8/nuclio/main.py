import base64
import json
import numpy as np

import yaml
import cv2
from modules import YOLOv8

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    context.user_data.labels = labels

    # Read the DL model
    model = YOLOv8("yolov8x.onnx", conf_thres=0.2, iou_thres=0.3)
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YoloV8 ONNX model")
    data = event.body
    buf = np.fromstring(base64.b64decode(data["image"]), np.uint8)
    threshold = float(data.get("threshold", 0.5))
    image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
    h, w, _ = image.shape

    boxes, scores, labels = context.user_data.model(image)

    results = []
    for label, score, box in zip(labels, scores, boxes):
        if score >= threshold:
            xtl = max(int(box[0]), 0)
            ytl = max(int(box[1]), 0)
            xbr = min(int(box[2]), w)
            ybr = min(int(box[3]), h)

            results.append({
                "confidence": str(score),
                "label": context.user_data.labels.get(label, "unknown"),
                "points": [xtl, ytl, xbr, ybr],
                "type": "rectangle",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
