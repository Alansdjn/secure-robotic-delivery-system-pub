import numpy as np
import random
from datetime import datetime

import torch
import xgboost as xgb

from core.dataset import SpokenDigitDataset
from core.model import SpokenDigitModel, FeatureExtractor
from core.data_preprocess import get_mel_img_data
from core.audio_util import AudioUtil


def generate_pins(length=4):
    digits = []
    for i in range(length):
        digits += [str(random.randint(0, 9))]
    return digits

# load CNN model
input_size = 3
cnn_model_name = './model/best/cnn3/best.ckpt'
ckpt = torch.load(cnn_model_name, map_location=torch.device('cpu'))
cnn_model = SpokenDigitModel(input_size=input_size).eval()
cnn_model.load_state_dict(ckpt['net_state_dict'])
feature_extractor = FeatureExtractor(cnn_model)

# load XGBoost model
xgb_model_name = './model/best/xgboost3/xgb.model'
xgb_model = xgb.Booster()
xgb_model.load_model(xgb_model_name)

total = 0
lr_pos_cnt = 0
xgb_pos_cnt = 0
next = 'y'
while next != 'n':
    print('>>'*20)

    # random generate 4 digits pin number
    pins = generate_pins()

    # record audio
    print('Please speak out the given pin numbers: [%s]' % ', '.join(pins))
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S--")
    file = "records/{}{}.wav".format(timestamp, ''.join(pins))
    # TODO
    print('TODO: >>>>>>>>>>>>>>>>>')
    file = './data/records/4286.wav'
    print('TODO: >>>>>>>>>>>>>>>>>')
    file = AudioUtils.arecord(file, duration=6)

    # split audio using a adaptive silence_thresh
    min_silence_len=200
    keep_silence=100
    seek_step=100
    audio_segment_files = AudioUtil.split_4_segments(file, min_silence_len, keep_silence, seek_step)

    # AudioUtil.waveplot(audio_segment_files, pins)

    # predict
    lr_pred_nums = []
    xgb_pred_nums = []
    for i, audio_file in enumerate(audio_segment_files):
        img = get_mel_img_data(audio_file)
        output = cnn_model(img.unsqueeze(0)).detach().numpy()
        lr_pred_num = np.argmax(output)
        lr_pred_num = ('unknown' if lr_pred_num == 10 else str(lr_pred_num))
        lr_pred_nums += [lr_pred_num]

        dtest = xgb.DMatrix(np.array(feature_extractor(img.unsqueeze(0))))
        xgb_pred_num = xgb_model.predict(dtest)
        xgb_pred_num = ('unknown' if int(xgb_pred_num[0]) == 10 else str(int(xgb_pred_num[0])))
        xgb_pred_nums += [xgb_pred_num]
        
    print('(CNN + LR) pred:     [%s] [%s]' % (', '.join(lr_pred_nums), 'match' if ''.join(lr_pred_nums) == ''.join(pins) else '-----'))
    print('(CNN + XGBoost) pred:[%s] [%s]' % (', '.join(xgb_pred_nums), 'match' if ''.join(xgb_pred_nums) == ''.join(pins) else '-----'))
    
    # continue or stop
    print('[%d] Continue? y/n' % total)
    next = input()











