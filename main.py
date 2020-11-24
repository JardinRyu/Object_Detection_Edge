import os
import sys
import cv2
import time
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

import datetime
import coco
import uff
import tensorrt as trt
import graphsurgeon as gs
#from config import model_ssd_inception_v2_coco_2017_11_17 as model
#from config import model_ssd_mobilenet_v1_coco_2018_01_28 as model
from config import model_ssd_mobilenet_v2_coco_2018_03_29 as model

ctypes.CDLL("lib/libflattenconcat.so")
COCO_LABELS = coco.COCO_CLASSES_LIST


# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record_org = False
record = False

# compile model into TensorRT
if not os.path.isfile(model.TRTbin):
    dynamic_graph = model.add_plugin(gs.DynamicGraph(model.path))
    uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.output_name, output_filename='tmp.uff')

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', model.dims)
        parser.register_output('MarkOutput_0')
        parser.parse('tmp.uff', network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(model.TRTbin, 'wb') as f:
            f.write(buf)


# create engine
with open(model.TRTbin, 'rb') as f:
    buf = f.read()
    engine = runtime.deserialize_cuda_engine(buf)


# create buffer
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    host_mem = cuda.pagelocked_empty(size, np.float32)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
context = engine.create_execution_context()


# inference
#TODO enable video pipeline
#TODO using pyCUDA for preprocess
capture = cv2.VideoCapture(0)
if (capture.isOpened() == False):
    print("NO CAMERA!")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
capture.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, img = capture.read()
    frame =  cv2.resize(img, (0, 0), fx=1, fy=1)
    #image = cv2.imread(sys.argv[1])
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model.dims[2],model.dims[1]))
    image = (2.0/255.0) * image - 1.0
    image = image.transpose((2, 0, 1))
    np.copyto(host_inputs[0], image.ravel())

    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    #print("execute times "+str(time.time()-start_time))

    output = host_outputs[0]
    height, width, channels = frame.shape
    for i in range(int(len(output)/model.layout)):
        prefix = i*model.layout
        index = int(output[prefix+0])
        label = int(output[prefix+1])
        conf  = output[prefix+2]
        xmin  = int(output[prefix+3]*width)
        ymin  = int(output[prefix+4]*height)
        xmax  = int(output[prefix+5]*width)
        ymax  = int(output[prefix+6]*height)

        if conf > 0.5:
            print("Detected {} with confidence {}".format(COCO_LABELS[label], "{0:.0%}".format(conf)))
            cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (0,0,255),3)
            cv2.putText(frame, COCO_LABELS[label],(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    elapsed_ms = (time.time() - start_time)
    if elapsed_ms != 0:
        fps = 1 / (elapsed_ms)
        str = 'FPS: %0.1f' % fps
        cv2.putText(frame, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("res", frame)
    
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    k = cv2.waitKey(1) & 0xff

    #Stop
    if k == ord('q') or k == 27:
        break
    #Capture (Original)
    elif k == ord('c'):
        print('Capture (Original)')
        cv2.imwrite("./" + now + ".png", img)
    #Capture (Detection)
    elif k == ord('C'):
        print('Capture (Detection)')
        cv2.imwrite("./" + now + ".png", frame)
    #Start Record (Original)
    elif k == ord('r'):
        print("Start Record (Original)")
        record_org = True
        video = cv2.VideoWriter("./" + now + ".avi", fourcc, 20.0, (img.shape[1], img.shape[0]))
    #Start Record (Detection)
    elif k == ord('R'):
        print("Start Record (Detection)")
        record = True
        video = cv2.VideoWriter("./" + now + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    #Stop Record
    elif k == ord('s'):
        print("Stop Record")
        record_org = False
        record = False
        
    if record_org == True:            
        video.write(img)
    if record == True:            
        video.write(frame)

capture.release()
cv2.destroyAllWindows()