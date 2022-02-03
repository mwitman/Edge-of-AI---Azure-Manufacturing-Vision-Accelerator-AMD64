from azure.iot.device import IoTHubModuleClient, Message
import cv2
import json

global module_client    
module_client = IoTHubModuleClient.create_from_edge_environment()
# module_client.connect()

class RouteHandler():

    def __init__():
        # module_client = IoTHubModuleClient.create_from_edge_environment()
        module_client.connect()

    def RetrainingSend(img_name, location, position, img_path):
        retrain_message = {
            'fs_name': 'training-images',
            'img_name': img_name,
            'location': location,
            'position': position,
            'path': img_path
        }
        retrain_str = json.dumps(retrain_message)
        try:
            message = Message(bytearray(retrain_str, 'utf-8'))
            module_client.send_message_to_output(message, "outputRetrainingSend")
            print("Message to outputRetrainingSend")

        except Exception as e:
            print ( "Unexpected error %s " % e )
            raise

    def AnnotatedSend(img_name, location, position, img_path):
        annotated_message = {
            'fs_name': 'annotated-images',
            'img_name': img_name,
            'location': location,
            'position': position,
            'path': img_path
        }
        annotated_str = json.dumps(annotated_message)
        try:
            message = Message(bytearray(annotated_str, 'utf-8'))
            module_client.send_message_to_output(message, "outputAnnotatedSend")
            print("Message to outputAnnotatedSend")

        except Exception as e:
            print ( "Unexpected error %s " % e )
            raise
    
    def FrameSend(img_name, location, position, img_path):
        frame_message = {
            'fs_name': 'raw-images',
            'img_name': img_name,
            'location': location,
            'position': position,
            'path': img_path
        }
        frame_str = json.dumps(frame_message)
        try:
            message = Message(bytearray(frame_str, 'utf-8'))
            module_client.send_message_to_output(message, "outputFrameSend")
            print("Message to outputFrameSend")

        except Exception as e:
            print ( "Unexpected error %s " % e )
            raise

    def InferenceSend(inference_msg):
        try:
            message = Message(bytearray(inference_msg, 'utf-8'))
            module_client.send_message_to_output(message, "outputInference")
            print("Inference sent.")

        except Exception as e:
            print ( "Unexpected error %s " % e )
            raise
            