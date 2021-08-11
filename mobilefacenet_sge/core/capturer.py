import os
import time
import shutil
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms

def extract_face(img, box, image_size=(112, 96), save_path=None):
    raw_image_size = img.size
    box = [
        int(max(box[0], 0)),
        int(max(box[1], 0)),
        int(min(box[2], raw_image_size[0])),
        int(min(box[3], raw_image_size[1])),
    ]

    face = img.crop(box).copy().resize(image_size, Image.BILINEAR)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        face.save(save_path)

    return face

def capture_detect_face(mtcnn, img=None, camera=None, saveroot='./data/capture'):
    max_capture_time = 10 # seconds
	# init save folder
    shutil.rmtree(saveroot)
    if not os.path.isdir(saveroot):
        os.makedirs(saveroot)

    if camera is not None:
        since = time.time()
        while True:
            # capture image
            ret, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # detect the largest faces
            boxes, probs, points = mtcnn.detect(img, landmarks=True)
            boxes, _, __ = mtcnn.select_boxes(boxes, probs, points, img, method=mtcnn.selection_method)
            if boxes is not None and len(boxes) > 0:
            	break
            elif (time.time() - since) <= max_capture_time:
                continue
            else:
                print('Do not capture any face from video in %d seconds' % max_capture_time)
                return None
    else:
        # detect the largest faces
        boxes, probs, points = mtcnn.detect(img, landmarks=True)
        boxes, _, __ = mtcnn.select_boxes(boxes, probs, points, img, method=mtcnn.selection_method)
        if boxes is None or len(boxes) == 0:
            print('Do not detect any face from given image data')
            return None

   	# extract the detected face
    tmp_extract_face_file = os.path.join(saveroot, 'extract_face.jpg')
    print('Save detected face: %s' % tmp_extract_face_file)
    face = extract_face(img, boxes[0], image_size=(96, 112), save_path=tmp_extract_face_file)
    # face = np.array(face).transpose((2, 0, 1)) # do not need to do this, do it in LFW

    # draw rectangle on the original image
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box.tolist(), outline='Red', width=2)
    file_name = '{}_{}.{}'.format(mtcnn.selection_method, datetime.now().strftime('%Y%m%d_%H%M%S'), 'jpg')
    capture_image_path = os.path.join(saveroot, file_name)
    print('Save captured image: %s' % capture_image_path)
    img.save(capture_image_path)
    
    return face

if __name__ == '__main__':
    mtcnn = MTCNN(thresholds=[0.5, 0.7, 0.7], selection_method='largest')

    # imgl = capture_detect_face(mtcnn, img=Image.open('./data/test/lfw-112X96/Wang_Wei/Wang_Wei_0002.jpg'))
    # print(type(imgl))
    # print(imgl.shape)

    # Init cv2
    cam = cv2.VideoCapture(0)
    print(cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)) # set video width
    print(cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)) # set video height
    imgr = capture_detect_face(mtcnn, camera=cam)
    print(type(imgr))
    print(imgr.shape)
