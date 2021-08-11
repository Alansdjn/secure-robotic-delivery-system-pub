import os
import torch
import torch.nn as nn
import numpy as np
import random
import xgboost as xgb
import shutil
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

from core.capturer import capture_detect_face
from core.model import MobileFacenet
from core.margin import ArcMarginProduct
from core.utils import init_log, train, test, save_model
from core.dataset import CASIA_Face, LFW
from core.evaluator import evaluation_10_fold

# load MobileFaceNet model
print('Loading pretrained MobileFaceNet model from [model/best/sge64-mfn/best.ckpt] ...')
mobilefacenet_model_name = 'model/best/sge64-mfn/best.ckpt'
ckpt = torch.load(mobilefacenet_model_name, map_location=torch.device('cpu'))
net = MobileFacenet().eval()
net.load_state_dict(ckpt['net_state_dict'])
print('    Best MobileFaceNet model epoch:', ckpt['epoch'])

# Init cv2
cam = cv2.VideoCapture(0)
# Init MTCNN, selection_method='center_weighted_size'
mtcnn = MTCNN(thresholds=[0.5, 0.7, 0.7], selection_method='largest')

# total = 0
# next = 'y'
# while next != 'n':
#     print('>>'*20)

#     imgl=Image.open('./data/test/lfw-112X96/Wang_Wei/Wang_Wei_0002.jpg')
#     facel = capture_detect_face(mtcnn, img=imgl)

#     print('Ready to take a photo? y/n')
#     next = input()
#     facer = capture_detect_face(mtcnn, camera=cam)
#     imgs = LFW.getitem(np.array(facel), np.array(facer))

#     res = [net(d.unsqueeze(0)).data.cpu().numpy() for d in imgs]
#     featureL = np.concatenate((res[0], res[1]), 1)
#     featureR = np.concatenate((res[2], res[3]), 1)

#     similarity = cosine_similarity(featureL, featureR)
#     print('similarity: ', similarity[0][0])

#     # continue or stop
#     total += 1
#     print('[%d] Continue? y/n' % total)
#     next = input()

###########
#debug
face_l_r_pair = [
    ['./data/test/lfw-112X96/Wang_Wei/Wang_Wei_0001.jpg', './data/test/lfw-112X96/Wang_Wei/Wang_Wei_0002.jpg'],
    ['./data/test/lfw-112X96/Wang_Wei/Wang_Wei_0002.jpg', './data/test/lfw-112X96/Wang_Wei/Wang_Wei_0003.jpg'],
    ['./data/test/lfw-112X96/Wang_Wei/Wang_Wei_0001.jpg', './data/test/lfw-112X96/Wang_Wei/Wang_Wei_0003.jpg'],
    ['./data/test/lfw-112X96/Wang_Wei/unknow_girl_0.jpg', './data/test/lfw-112X96/Wang_Wei/Wang_Wei_0001.jpg'],
    ['./data/test/lfw-112X96/Wang_Wei/unknow_girl_0.jpg', './data/test/lfw-112X96/Wang_Wei/Wang_Wei_0002.jpg'],
    ['./data/test/lfw-112X96/Wang_Wei/unknow_girl_0.jpg', './data/test/lfw-112X96/Wang_Wei/Wang_Wei_0003.jpg'],
]

for pair in face_l_r_pair:
    print('>>'*20)

    imgl=Image.open(pair[0])
    facel = capture_detect_face(mtcnn, img=imgl)

    imgr=Image.open(pair[1])
    facer = capture_detect_face(mtcnn, img=imgr)
    imgs = LFW.getitem(np.array(facel), np.array(facer))

    res = [net(d.unsqueeze(0)).data.cpu().numpy() for d in imgs]
    featureL = np.concatenate((res[0], res[1]), 1)
    featureR = np.concatenate((res[2], res[3]), 1)

    similarity = cosine_similarity(featureL, featureR)
    print(pair)
    print('similarity: ', similarity[0][0])

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()








