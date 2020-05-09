#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
import time
import pandas as pd 

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("-v","--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("-p","--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("-w","--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("-t","--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

#Load the Caffe model 
#net = cv2.dnn.readNetFromTensorflow(args.prototxt, args.weights)
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
data_list = []

while True:
    data_capture = {}
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size. 
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(frame_resized, 1/127.5, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    inf_start_time = time.time() # time inference start
    detections = net.forward()
     

    #Size of frame resize (300x300)
    widthFactor = frame_resized.shape[1]
    HeightFactor = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
            inf_end_time = time.time() # time inference ended
            diff_time = abs(inf_start_time - inf_end_time) # difference in time



            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * widthFactor) 
            yLeftBottom = int(detections[0, 0, i, 4] * HeightFactor)
            xRightTop   = int(detections[0, 0, i, 5] * widthFactor)
            yRightTop   = int(detections[0, 0, i, 6] * HeightFactor)
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)

            inf_message = "Inference Time: {:.3f}s".format(diff_time)
            # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
            cv2.putText(frame, inf_message, (15,15), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (200, 10, 10), 1)


            data_capture['time'] = time.strftime("%H:%M:%S", time.localtime())
            data_capture['confidence']= confidence
            data_capture['inference_time'] = diff_time

            data_list.append(data_capture)

            # Draw label and confidence of prediction in frame resized        
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(round(confidence,3)) + " , inference time = " + str(round(diff_time,3))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label) #print class and confidence
            
            

    df = pd.DataFrame(data_list)
    df.to_csv("\results\results.csv")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break

