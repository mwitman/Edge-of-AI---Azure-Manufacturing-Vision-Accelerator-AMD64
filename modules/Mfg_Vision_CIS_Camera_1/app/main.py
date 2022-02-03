import os
import sys
from time import sleep

capturing = False

class CaptureInferenceStore():

    def __init__(self, camGvspAllied, camGvspBasler, camRTSP, camFile, camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, modelACV,
                modelFile, labelFile, targetDim, probThres, iouThres, retrainInterval, storeRawFrames, storeAllInferences, SqlDb, SqlPwd):

        modelPath = os.path.join('/model_volume/',modelFile)
        labelPath = os.path.join('/model_volume/',labelFile)
        sleep(5)
        if modelACV:
            from inference.onnxruntime_predict import initialize_acv
            initialize_acv(modelPath, labelPath)
        else:
            from inference.onnxruntime_yolov5 import initialize_yolov5
            initialize_yolov5(modelPath, labelPath, targetDim, probThres, iouThres, camLocation, camPosition)
        sleep(1)

        if camGvspAllied:     
            from capture.allied.camera_gvsp_allied import Allied_GVSP_Camera
            Allied_GVSP_Camera(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, modelACV, modelFile, labelFile, 
                targetDim, probThres, iouThres, retrainInterval, SqlDb, SqlPwd, storeRawFrames, storeAllInferences)

        if camGvspBasler:     
            from capture.basler.camera_gvsp_basler import Basler_GVSP_Camera
            Basler_GVSP_Camera(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, modelACV, modelFile, labelFile, 
                targetDim, probThres, iouThres, retrainInterval, SqlDb, SqlPwd, storeRawFrames, storeAllInferences)
            
        elif camRTSP:
            from capture.RTSP.camera_rtsp import RTSP_Camera
            RTSP_Camera(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, modelACV, modelFile, labelFile, 
                targetDim, probThres, iouThres, retrainInterval, SqlDb, SqlPwd, storeRawFrames, storeAllInferences)

        elif camFile:
            from capture.file.camera_file import Cam_File_Sink
            Cam_File_Sink(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, modelACV, modelFile, labelFile, 
                targetDim, probThres, iouThres, retrainInterval, SqlDb, SqlPwd, storeRawFrames, storeAllInferences)

        else:
            print("No camera found")
    
def __convertStringToBool(env: str) -> bool:
    if env in ['true', 'True', 'TRUE', '1', 'y', 'YES', 'Y', 'Yes']:
        return True
    elif env in ['false', 'False', 'FALSE', '0', 'n', 'NO', 'N', 'No']:
        return False
    else:
        raise ValueError('Could not convert string to bool.')
       
if __name__ == "__main__":

    try:
        CAMERA_GVSP_ALLIED = __convertStringToBool(os.environ["CAMERA_GVSP_ALLIED"])
        CAMERA_GVSP_BASLER = __convertStringToBool(os.environ["CAMERA_GVSP_BASLER"])
        CAMERA_RTSP = __convertStringToBool(os.environ["CAMERA_RTSP"])
        CAMERA_FILE = __convertStringToBool(os.environ["CAMERA_FILE"])
        CAMERA_ID = os.environ["CAMERA_ID"]
        CAMERA_TRIGGER = __convertStringToBool(os.environ["CAMERA_TRIGGER"])
        CAMERA_URI = os.environ["CAMERA_URI"]
        CAMERA_LOCATION = os.environ["CAMERA_LOCATION"]
        CAMERA_POSITION = os.environ["CAMERA_POSITION"]
        CAMERA_FPS = float(os.environ["CAMERA_FPS"])
        INFERENCE_FPS = float(os.environ["INFERENCE_FPS"])
        MODEL_ACV = __convertStringToBool(os.environ["MODEL_ACV"])
        MODEL_FILE = os.environ["MODEL_FILE"]
        LABEL_FILE = os.environ["LABEL_FILE"]
        TARGET_DIM = int(os.environ["TARGET_DIM"])
        PROB_THRES = float(os.environ["PROB_THRES"])
        IOU_THRES = float(os.environ["IOU_THRES"])
        RETRAIN_INTERVAL = int(os.environ["RETRAIN_INTERVAL"])
        STORE_RAW_FRAMES = __convertStringToBool(os.environ["STORE_RAW_FRAMES"])
        STORE_ALL_INFERENCES = __convertStringToBool(os.environ["STORE_ALL_INFERENCES"])
        MSSQL_DB = os.environ["MSSQL_DB"]
        MSSQL_PWD = os.environ["MSSQL_PWD"]

    except ValueError as error:
        print(error)
        sys.exit(1)

    CaptureInferenceStore(CAMERA_GVSP_ALLIED, CAMERA_GVSP_BASLER, CAMERA_RTSP, CAMERA_FILE, CAMERA_ID, CAMERA_TRIGGER, CAMERA_URI, CAMERA_LOCATION, CAMERA_POSITION, CAMERA_FPS, 
            INFERENCE_FPS, MODEL_ACV, MODEL_FILE, LABEL_FILE, TARGET_DIM, PROB_THRES, IOU_THRES, RETRAIN_INTERVAL, STORE_RAW_FRAMES, STORE_ALL_INFERENCES, MSSQL_DB, MSSQL_PWD)
    
