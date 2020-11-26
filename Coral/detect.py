import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import tflite_runtime.interpreter as tflite
import time
import datetime

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    record_org = False
    record = False

    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        frame =  cv2.resize(image, (0, 0), fx=1, fy=1)
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        starttime = time.time()

        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        cv2_im = append_objs_to_img(cv2_im, objs, labels)
        elapsed_ms = (time.time() - starttime)

        if elapsed_ms != 0:
            fps = 1 / (elapsed_ms)
            str = 'FPS: %0.1f' % fps
            cv2.putText(frame, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)    

        cv2.imshow('frame', cv2_im)

        now = datetime.datetime.now().strftime("%d_%H-%M-%S")
        k = cv2.waitKey(1) & 0xff

        #Stop
        if k == ord('q') or k == 27:
            break
        #Capture (Original)
        elif k == ord('c'):
            print('Capture (Original)')
            cv2.imwrite("./" + now + ".png", image)
        #Capture (Detection)
        elif k == ord('C'):
            print('Capture (Detection)')
            cv2.imwrite("./" + now + ".png", cv2_im)
        #Start Record (Original)
        elif k == ord('r'):
            print("Start Record (Original)")
            record_org = True
            video = cv2.VideoWriter("./" + now + ".avi", fourcc, 20.0, (image.shape[1], image.shape[0]))
        #Start Record (Detection)
        elif k == ord('R'):
            print("Start Record (Detection)")
            record = True
            video = cv2.VideoWriter("./" + now + ".avi", fourcc, 20.0, (cv2_im.shape[1], cv2_im.shape[0]))
        #Stop Record
        elif k == ord('s'):
            print("Stop Record")
            record_org = False
            record = False
        
        if record_org == True:            
            video.write(image)
        if record == True:            
            video.write(cv2_im)

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
