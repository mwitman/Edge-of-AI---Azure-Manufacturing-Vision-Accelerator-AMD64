import numpy as np
import onnxruntime as ort
import json
from objdict import ObjDict
import cv2
import time
import os
from datetime import datetime
from pathlib import Path
from PIL import Image 
import torch
from inference.utils.general import non_max_suppression
import cv2 # for annotations
from capture.frame_save import FrameSave
from route_handler import RouteHandler

providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

class ONNXRuntimeObjectDetection():

    def __init__(self, model_path, labels, target_dim, target_prob, target_iou, cam_location, cam_position):
        self.target_dim = target_dim
        self.target_prob = target_prob
        self.target_iou = target_iou
        self.camLocation = cam_location
        self.camPosition = cam_position
        
        self.device_type = ort.get_device()
        print(f"ORT device: {self.device_type}")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # Uncomment lines below to check the shape of the model inputs if needed
        # self.output_name = self.session.get_outputs()[0].name
        # self.batch_size = self.session.get_inputs()[0].shape[0]
        # self.channels = self.session.get_inputs()[0].shape[1]
        # self.img_size_h = self.session.get_inputs()[0].shape[2]
        # self.img_size_w = self.session.get_inputs()[0].shape[3]
        # print("Input: {} Output: {} Batch Size: {} Model ImgH: {} Model ImgW: {}".format(self.input_name,self.output_name,self.batch_size,self.img_size_h,self.img_size_w))
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        self.labels = labels
        self.num_classes = len(labels)
             
    def predict(self, pp_image, image):
        inputs = pp_image
        if self.is_fp16:
            inputs = inputs.astype(np.float16)
        outputs = self.session.run(None, {self.input_name: inputs})
        # Uncomment lines below to check the shape of the model outputs if needed
        # print('Concatenated outputs shape: {}'.format(outputs[0].shape))
        # print('Separated outputs shape: {}, {}, {}'.format(outputs[1].shape, outputs[2].shape, outputs[3].shape))

        filterd_predictions = non_max_suppression(torch.tensor(outputs[0]), conf_thres = self.target_prob, iou_thres = self.target_iou)
        # Prints the raw prediction from the output tensor
        # print(filterd_predictions) 
        img = image

        predictions = []
        ONNXRuntimeObjectDetection.inference_count = 0

        try:
            for pred in filterd_predictions[0]: 
                x1 = round(float(pred[0]),8)
                y1 = round(float(pred[1]),8)
                x2 = round(float(pred[2]),8)
                y2 = round(float(pred[3]),8)
                probability = round(float(pred[4]),8)
                labelId = int(pred[5])
                labelName = str(self.labels[labelId])

                pred = ObjDict()
                pred.probability = float(probability*100)
                pred.labelId = int(labelId)
                pred.labelName = labelName
                pred.bbox = {
                    'left': x1,
                    'top': y1,
                    'width': x2,
                    'height': y2
                }
                predictions.append(pred)

            return predictions

        except:
            print("No predictions present")    

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def checkModelExtension(fp):
  ext = os.path.splitext(fp)[-1].lower()
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def initialize_yolov5(model_path, labels_path, target_dim, target_prob, target_iou, cam_location, cam_position):
    print('Loading labels...\n', end='')
    checkModelExtension(model_path)
    with open(labels_path, 'r') as f:
        labels = [l.strip() for l in f.readlines()]    
    print('Loading model...\n', end='')
    global ort_model
    ort_model = ONNXRuntimeObjectDetection(model_path, labels, target_dim, target_prob, target_iou, cam_location, cam_position)
    print('Success!')

def predict_yolo(image):
    log_msg('Predicting image')
    frame = np.asarray(image)
    frame = frame.astype(np.float32)
    frame = frame.transpose(2,0,1)
    frame = np.expand_dims(frame, axis=0)
    frame /= 255.0 # normalize pixels
    # print(f"Batch-Size, Channel, Height, Width : {frame.shape}")
    t1 = time.time()
    predictions = ort_model.predict(frame, image)
    t2 = time.time()
    t_infer = (t2-t1)*1000
    response = {
        'created': datetime.utcnow().isoformat(),
        'inference_time': t_infer,
        'predictions': predictions
        }
    return response

def warmup_image(batch_size, warmup_dim):
    for _ in range(batch_size):
        yield np.zeros([warmup_dim, warmup_dim, 3], dtype=np.uint8)

