"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np 


import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from collections import deque

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### DONE: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # intialising variables 
    image_flag = False 
    current_request_id = 0 
    last = 0 
    total = 0 
    start = 0 


    max_len = 30
    track_threshold = 0.1 
    track  = deque(maxlen=max_len)

    # Initialise the class
    network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### DONE: Load the model through `network` ###
    
    n, c, h, w = network.load_model(args.model, args.device,current_request_id,
                                     args.cpu_extension)[1]

    


    ### DONE: Handle the input stream ###
    # check for CAM
    if args.input == 'CAM':
        input_stream = 0 
    # check for image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True 
        input_stream = args.input
    # check for video 
    else:
        input_stream = args.input

    cap = cv2.VideoCapture(input_stream) 
    
    if input_stream:
        cap.open(args.input)

    initial_w = int(cap.get(3))
    initial_h = int(cap.get(4))



    ### DONE: Loop until stream is over ###
    while cap.isOpened():
        ### DONE: Read from the video capture ###   
        flag, frame = cap.read() 
        if not flag:
            break 
        key_pressed = cv2.waitKey(60)

        ### DONE: Pre-process the image as needed ###
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2,0,1))
        image = image.reshape((n, c, h, w))

        ### DONE: Start asynchronous inference for specified request ###
        inf_start_time = time.time() # time inference start
        network.exec_net(current_request_id, image)



        ### DONE: Wait for the result ###
        if network.wait(current_request_id) == 0:
            inf_end_time = time.time() # time inference ended
            diff_time = abs(inf_start_time - inf_end_time) # difference in time 

        ### DONE: Get the results of the inference request ###
        result = network.get_output(current_request_id)  


        ### DONE: Extract any desired stats from the results ###
        frame, current_count = extract(frame, result, prob_threshold,
                                    initial_w, initial_h)

        inf_message = "Inference Time: {:.3f}ms".format(diff_time)

        cv2.putText(frame, inf_message, (15,15), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (200, 10, 10), 1)
            

        ### DONE: Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        track.append(current_count)
        num_tracked = 0 
        if np.sum(track) / max_len > track_threshold:
            num_tracked = 1


        if num_tracked > last:
            global start_time
            start_time = time.time()
            total = total + current_count - last 
            client.publish("person", json.dumps({"total":total}))

        if num_tracked < last:
            end_time = time.time()
            duration = int(end_time - start_time)
            client.publish("person/duration", json.dumps({"duration":duration}))

        client.publish("person",json.dumps({"count":current_count}))
        last = num_tracked

        if key_pressed == 27:
            break

        ### DONE: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### DONE: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite("output_image.jpeg", frame)

    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect() 



def extract(frame, result,prob_threshold,initial_w, initial_h):
    current_count =0 
    for object in result[0][0]:
        if object[2] > prob_threshold:
            xmin = int(object[3] * initial_w)
            ymin = int(object[4] * initial_h)
            xmax = int(object[5] * initial_w)
            ymax = int(object[6] * initial_h)
            cv2.rectangle(frame, (xmin,ymin),(xmax, ymax), (0, 55, 255),1)
            current_count += 1 

    return frame, current_count


def main():
    """
    Load the network and parse the output.

    :return: None
    """

    # global initial_w, initial_h,prob_threshold

    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
