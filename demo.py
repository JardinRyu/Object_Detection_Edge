#!/usr/bin/python3
import numpy as np
import cv2
import jetson.inference
import jetson.utils
 
# setup the network we are using
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video.mp4')
cap.set(3,640)
cap.set(4,480)
 
while (True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    w = frame.shape[1]
    h = frame.shape[0]
    # to RGBA
    # to float 32
    input_image = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA).astype(np.float32)
    # move the image to CUDA:
    input_image = jetson.utils.cudaFromNumpy(input_image)
 
    detections = net.Detect(input_image, w, h)
    count = len(detections)
    print("detected {:d} objects in image".format(len(detections)))
    for detection in detections:
        print(detection)
 
    # print out timing info
    net.PrintProfilerTimes()
    # Display the resulting frame
    numpyImg = jetson.utils.cudaToNumpy(input_image, w, h, 4)
    # now back to unit8
    result = numpyImg.astype(np.uint8)
    # Display fps
    fps = 1000.0 / net.GetNetworkTime()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    cv2.putText(result, "FPS: " + str(int(fps)) + ' | Detecting', (11, 20), font, 0.5, (32, 32, 32), 4, line)
    cv2.putText(result, "FPS: " + str(int(fps)) + ' | Detecting', (10, 20), font, 0.5, (240, 240, 240), 1, line)
    cv2.putText(result, "Total: " + str(count), (11, 45), font, 0.5, (32, 32, 32), 4, line)
    cv2.putText(result, "Total: " + str(count), (10, 45), font, 0.5, (240, 240, 240), 1, line)
    # show frames
    cv2.imshow('frame', result)
    if cv2.waitKey(1) &amp; 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
