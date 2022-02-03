import os
import sys
import json
import tempfile
from datetime import datetime
from urllib.request import urlopen
import time
import io

import onnxruntime
import onnx

import numpy as np
from PIL import Image, ImageDraw

from inference.onnxruntime_object_detection import ObjectDetection

MODEL_FILENAME = os.path.join('/model_volume/',os.environ["MODEL_FILE"])
LABELS_FILENAME = os.path.join('/model_volume/',os.environ["LABEL_FILE"])

providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(model_filename))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        print("Model pre-loaded on initialize - complete")
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def initialize_acv(modelPath, labelPath):
    """Load labels and onnx model"""
    print('Loading labels...', end='')
    with open(labelPath, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    print("{} found. Success!".format(len(labels)))
    
    print('Loading model...', end='')
    global od_model
    od_model = ONNXRuntimeObjectDetection(modelPath, labels)
    print('Success!')

def predict_acv(image):
    log_msg('Predicting image')
    w, h = image.size
    log_msg("Image size: {}x{}".format(w, h))
    t1 = time.time()
    predictions = od_model.predict_image(image)
    t2 = time.time()
    t_infer = (t2-t1)*1000
    response = predictions
    response = {
                'created': datetime.utcnow().isoformat(),
                'inference_time': t_infer,
                'predictions': predictions
                }
    
    return response
