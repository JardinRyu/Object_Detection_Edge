import numpy as np
import cv2
import jetson.inference
import jetson.utils
import datetime
 
# setup the network we are using
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record_org = False
record = False

capture = cv2.VideoCapture(0)
if (capture.isOpened() == False):
    print("NO CAMERA!")
capture.set(3, CAMERA_WIDTH)
capture.set(4, CAMERA_HEIGHT)

while (True):
    ret, frame = capture.read()
    
    w = frame.shape[1]
    h = frame.shape[0]
    # to RGBA
    # to float 32
    input_image = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA).astype(np.float32)
    # move the image to CUDA:
    input_image = jetson.utils.cudaFromNumpy(input_image)
 
    detections = net.Detect(input_image, w, h)
    count = len(detections)
    
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

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    k = cv2.waitKey(1) & 0xff
    
    if k == ord('q') or k == 27:
            break
    #Capture (Original)
    elif k == ord('c'):
        print('Capture (Original)')
        cv2.imwrite("./" + now + ".png", frame)
    #Capture (Detection)
    elif k == ord('C'):
        print('Capture (Detection)')
        cv2.imwrite("./" + now + ".png", result)
    #Start Record (Original)
    elif k == ord('r'):
        print("Start Record (Original)")
        record_org = True
        video = cv2.VideoWriter("./" + now + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    #Start Record (Detection)
    elif k == ord('R'):
        print("Start Record (Detection)")
        record = True
        video = cv2.VideoWriter("./" + now + ".avi", fourcc, 20.0, (result.shape[1], result.shape[0]))
    #Stop Record
    elif k == ord('s'):
        print("Stop Record")
        record_org = False
        record = False
        
    if record_org == True:            
        video.write(frame)
    if record == True:            
        video.write(result)
 
# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
