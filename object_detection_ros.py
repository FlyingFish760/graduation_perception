import roslibpy
import numpy as np

from ultralytics import YOLO
import cv2



def image_callback(message):
    # Convert ROS string data to numpy array
    # Decode string to bytes
    # data_bytes = message['data']
    # img_array = np.frombuffer(data_bytes, dtype=np.uint8)
    img_array = np.array(message['data'], dtype=np.uint8)

    

    height = message['height']
    width = message['width']
    channels = 3

    # Trim array if it has extra bytes
    expected_size = height * width * 3
    if img_array.size > expected_size:
        img_array = img_array[:expected_size]

    img = img_array.reshape((height, width, channels))
    
    cv2.imshow('Received Image', img)
    cv2.waitKey(1)
    
    # Run object detection
    detections = detect_objects(img)
    
    # publish detection results to ROS
    detection_msg = {'objects': detections}
    publisher.publish(roslibpy.Message({'data': str(detection_msg)}))

def detect_objects(img):
    return [{'label':'car','x':100,'y':200,'w':50,'h':50}]

# Connect to ROSBridge
target_ip = '192.168.56.128'
client = roslibpy.Ros(host=target_ip, port=9090)
client.run()
print("client is connected:", client.is_connected)

# set Subscribers
subscriber = roslibpy.Topic(client, '/image', 'sensor_msgs/Image')
subscriber.subscribe(image_callback)

# set Publishers
publisher = roslibpy.Topic(client, '/detections', 'std_msgs/String')

try:
    while True:
        pass
except KeyboardInterrupt:
    subscriber.unsubscribe()
    client.terminate()
    print("keyboard interruption")
