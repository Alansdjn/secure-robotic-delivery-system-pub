import numpy as np
import random
from datetime import datetime
import time
import socket
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn

import xgboost as xgb
from facenet_pytorch import MTCNN

from cnn_xgboost.core.model import SpokenDigitModel, FeatureExtractor
from cnn_xgboost.core.audio_util import AudioUtil
from cnn_xgboost.core.data_preprocess import get_mel_img_data
from mobilefacenet_sge.core.capturer import capture_detect_face
from mobilefacenet_sge.core.model import MobileFacenet
from mobilefacenet_sge.core.dataset import LFW
from speaker_verification_ghostnet.ghostnet import ghostnet
from speaker_verification_ghostnet.data_preprocess import cal_melspectrogram_features


def timeout(delivery_expire_time):
    return (datetime.now()-delivery_expire_time).total_seconds() <= 0

def load_pins():
    ipaddr = '192.168.31.194'
    port = 12345
    print('Connect to server %s:%s ...' % (ipaddr, port))
    s = socket.socket()
    s.connect((ipaddr, port))
    pins = s.recv(1024).decode()
    print('Receive PIN numbers from the server:', pins)
    return pins.split(',')


def load_info():
    pins = load_pins()
    customer_face_img = './mobilefacenet_sge/data/test/lfw-112X96/Wang_Wei/Wang_Wei_0001.jpg'
    # customer_face_img = './mobilefacenet_sge/data/test/lfw-112X96/Wang_Wei/unknow_girl_0.jpg'
    customer_audio = './speaker_verification_ghostnet/dataset/enrollment/wangwei_1934.wav'
    delivery_expire_time = datetime.strptime('2020-07-28 15:00:00', "%Y-%m-%d %H:%M:%S")
    return pins, customer_face_img, customer_audio, delivery_expire_time


def predict_pins(feature_extractor, xgb_model, file):
    # split audio using a adaptive silence_thresh
    audio_segment_files = AudioUtil.split_4_segments(file, min_silence_len=200, keep_silence=100, seek_step=100)

    # predict
    xgb_pred_nums = []
    for i, audio_file in enumerate(audio_segment_files):
        img = get_mel_img_data(audio_file)
        dtest = xgb.DMatrix(np.array(feature_extractor(img.unsqueeze(0))))
        pred_num = xgb_model.predict(dtest)
        xgb_pred_nums.append(str(int(pred_num[0])))

    return xgb_pred_nums


def cal_voiceprint_cosine_similarity(embedder_net, record_audio_path, enrolled_audio_path):
    enrollment_batch = torch.tensor(cal_melspectrogram_features(record_audio_path))
    verification_batch = torch.tensor(cal_melspectrogram_features(enrolled_audio_path))

    enrollment_embeddings = embedder_net(enrollment_batch)
    verification_embeddings = embedder_net(verification_batch)

    enrollment_embeddings = torch.mean(enrollment_embeddings, 0)
    verification_embeddings = torch.mean(verification_embeddings, 0)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(enrollment_embeddings.view(1, -1), verification_embeddings.view(1, -1))
    return output


def cooperative_person_identification(feature_extractor, xgb_model, ghostnet_model, pins, 
                                        enrolled_audio_path, thresholds = 0.5):
    pins_match, person_verified = None, None

    # record audio
    print('Please tell me your pin numbers')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    record_audio_path = "./cnn_xgboost/data/records/{}{}.wav".format(timestamp, ''.join(pins))
    print('Ready? (Press enter to continue)')
    next = input()
    record_audio_path = AudioUtil.arecord(record_audio_path, duration=6)

    # predict PINs
    print('Predict PINs using recorded audio, record audio path: %s' % record_audio_path)
    pred_pins = predict_pins(feature_extractor, xgb_model, record_audio_path)
    print('Predicted PINs is: %s' % ' '.join(pred_pins))

    pins_match = (''.join(pred_pins) == ''.join(pins))
    if not pins_match:
        print('Predicted PINs is not correct.')
        return pins_match, person_verified
    print('Predicted PINs is correct.')
    
    # check person
    print('Calculate cosine similarity between %s and %s ...' % (record_audio_path, enrolled_audio_path))
    cos_sim = cal_voiceprint_cosine_similarity(ghostnet_model, record_audio_path, enrolled_audio_path)
    print('The cosine similarity is %.2f, threshold is %.2f.' % (cos_sim, thresholds))
    person_verified = (cos_sim > thresholds)
    
    return pins_match, person_verified


def non_cooperative_person_identification(mtcnn, mfn_model, img_path, camera, thresholds = 0.5):
    count_down_time = 3
    while count_down_time > 0:
        print('Robot will start to capture your face %d seconds later ...' % count_down_time)
        time.sleep(1)
        count_down_time -= 1

    print('Capture and detect face from video stream ...')
    facer = capture_detect_face(mtcnn, camera=camera, saveroot='./mobilefacenet_sge/data/capture')
    if facer is None:
        return False

    print("Load customer's registered face features ...")
    identified_person_img=Image.open(img_path)
    facel = capture_detect_face(mtcnn, img=identified_person_img, saveroot='./mobilefacenet_sge/data/capture')

    imgs = LFW.getitem(np.array(facel), np.array(facer))

    # generate left face features
    res = [mfn_model(d.unsqueeze(0)).data.cpu().numpy() for d in imgs[:2]]

    print("Generate detected face features ...")
    res += [mfn_model(d.unsqueeze(0)).data.cpu().numpy() for d in imgs[2:]]

    featureL = np.concatenate((res[0], res[1]), 1)
    featureR = np.concatenate((res[2], res[3]), 1)

    print('Calculate cosine similarity between %s and captured face ...' % (img_path))
    similarity = cosine_similarity(featureL, featureR)
    print('The cosine similarity is %.2f, threshold is %.2f.' % (similarity[0][0], thresholds))

    return True if similarity > thresholds else False


if __name__ == '__main__':
    print('Robot start ...')
    device = torch.device('cpu')

    # load CNN model
    print('>'*40)
    print('Initialize SpokenDigitModel ...')
    cnn_model_name = './cnn_xgboost/model/best/cnn3/best.ckpt'
    cnn_model = SpokenDigitModel(input_size=3).eval()
    cnn_model.load_state_dict(torch.load(cnn_model_name, map_location=device)['net_state_dict'])
    feature_extractor = FeatureExtractor(cnn_model)

    # load XGBoost model
    print('Initialize XGBoost model ...')
    xgb_model_name = './cnn_xgboost/model/best/xgboost3/xgb.model'
    xgb_model = xgb.Booster(model_file=xgb_model_name)
    xgb_model.set_param({'predictor': 'cpu_predictor'})

    # load ghostnet model, for voiceprint feature extract
    print('Initialize Ghostnet model ...')
    ghostnet_model_name = './speaker_verification_ghostnet/speech_id_checkpoint/model.model'
    ghostnet_model = ghostnet(inp=1, num_classes=512).eval()
    ghostnet_model.load_state_dict(torch.load(ghostnet_model_name, map_location=device)['embedder_net_dict'])

    # load MobileFacenet model
    print('Initialize MobileFacenet model ...')
    mobilefacenet_model_name = './mobilefacenet_sge/model/best/sge64-mfn/best.ckpt'
    mfn_model = MobileFacenet().eval()
    mfn_model.load_state_dict(torch.load(mobilefacenet_model_name, map_location=device)['net_state_dict'])

    # Init cv2
    print('Initialize camera ...')
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print('Open camera fail.')
        exit(0)

    # Init MTCNN, selection_method='center_weighted_size'
    print('Initialize MTCNN ...')
    mtcnn = MTCNN(thresholds=[0.5, 0.7, 0.7], selection_method='largest')

    print('Load delivery information (PINs, enrolled customer face features and voiceprint features) ...')
    pins, face_path, audio_path, delivery_expire_time = load_info()

    delivery_complete, delivery_cancel = False, False

    while not timeout(delivery_expire_time):
        print('>'*40)
        max_retry_times = 3
        while max_retry_times > 0:
            pins_match, person_verified = cooperative_person_identification(feature_extractor, 
                                                                            xgb_model, 
                                                                            ghostnet_model, 
                                                                            pins, 
                                                                            audio_path)
            if pins_match and person_verified:
                print('Delivery success')
                delivery_complete = True
                break

            if pins_match and not person_verified:
                print('PIN code is correct, person identification fail, delivery cancel')
                delivery_cancel = True
                break

            max_retry_times -= 1
            print('Cooperative person identification fail, you can retry %d more times' % max_retry_times) 

        if delivery_complete or delivery_cancel:
            break

        print('>'*40)
        print('You have tried too many times, we must check your identification using facial verification ...')
        find_customer = False
        while not timeout(delivery_expire_time):
            find_customer = non_cooperative_person_identification(mtcnn, mfn_model, face_path, cam)
            if find_customer:
                print('Person identified, switch to cooperative mode ...')
                break
            print('Person identification failed, retry ...')

        if timeout(delivery_expire_time):
            print('Delivery timeout, delivery cancel')
            break

    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv2.destroyAllWindows()








