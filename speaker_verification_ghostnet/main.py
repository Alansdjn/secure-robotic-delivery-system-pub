import os
import random
import time

import numpy as np
import librosa
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hparam import hparam as hp
from dataset import TimitSpeakerDataset
from ge2e_loss import GE2ELoss
from ghostnet import ghostnet
from utils import get_centroids, get_cossim
from data_preprocess import cal_melspectrogram_mfcc

def extract_features(path):
    # load utterance audio
    utter, sr = librosa.core.load(path, hp.data.sr)
    # trim silent edges
    enrolled_utter, _ = librosa.effects.trim(utter, top_db=15)
    # sf.write('./dataset/enrollment/1934-trim.wav', enrolled_utter, sr)
    print(len(utter), len(enrolled_utter))

    if len(enrolled_utter) < sliding_window_length:
        print('\tutterance, length: %d, does not have enough non-slience audio. Padding it ...' % len(utter))
        enrolled_utter = librosa.util.pad_center(enrolled_utter, sliding_window_length)

    utterances_spec = []
    stop = False
    for start in range(0, len(enrolled_utter), sliding_stride):
        end = start+sliding_window_length
        if end >= len(enrolled_utter):
            start = len(enrolled_utter)-sliding_window_length
            end = len(enrolled_utter)
            stop = True

        # cal mel spec and mfcc features
        utter_part = enrolled_utter[start:end]
        print('(%d, %d)' % (start, end))
        mel_spec, mfccs = cal_melspectrogram_mfcc(utter_part)

        if hp.model.input_size == 2:
            features = np.array([mel_spec, mfccs])
        else:
            features = np.array([mel_spec])

        utterances_spec.append(features)

        if stop:
            break

    return np.array(utterances_spec)

# lower bound of utterance length
sliding_window_length = int((hp.data.tisv_frame-1) * hp.data.hop * hp.data.sr)
sliding_stride = int(sliding_window_length//2)
win_length=int(hp.data.window * hp.data.sr)
hop_length=int(hp.data.hop * hp.data.sr)
print("utterrance sliding window length: %d, sliding stride: %d, mel window length: %d, mel hop length: %d" 
        % (sliding_window_length, sliding_stride, win_length, hop_length))

first_speaker_path = './dataset/enrollment/SA2.wav'
second_speaker_path = './dataset/enrollment/2102.wav'

f1 = extract_features(first_speaker_path)
f2 = extract_features(second_speaker_path)

print(f1.shape)
print(f2.shape)

device = torch.device('cpu')
enrollment_batch = torch.tensor(f1).to(device)
verification_batch = torch.tensor(f2).to(device)

print('Load model from: %s' % hp.model.model_path)
embedder_net = ghostnet(inp=hp.model.input_size, num_classes=hp.model.feature_size).to(device)
embedder_net.load_state_dict(torch.load(hp.model.model_path, map_location=device)['embedder_net_dict'])
embedder_net.eval()

enrollment_embeddings = embedder_net(enrollment_batch)
verification_embeddings = embedder_net(verification_batch)

print(enrollment_embeddings.shape)
print(verification_embeddings.shape)

enrollment_embeddings = torch.mean(enrollment_embeddings, 0)
verification_embeddings = torch.mean(verification_embeddings, 0)

print(enrollment_embeddings.shape)
print(verification_embeddings.shape)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(enrollment_embeddings.view(1, -1), verification_embeddings.view(1, -1))

print(output)

# enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
# verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))



















