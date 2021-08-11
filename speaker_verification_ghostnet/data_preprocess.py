#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
import shutil

if __name__ == '__main__':
    from hparam import hparam as hp
elif __name__ == 'speaker_verification_ghostnet.data_preprocess':
    from speaker_verification_ghostnet.hparam import hparam as hp


def cal_melspectrogram_features(path, top_db=15):
    # lower bound of utterance length
    sliding_window_length = int((hp.data.tisv_frame-1) * hp.data.hop * hp.data.sr)
    sliding_stride = int(sliding_window_length//2)

    # load utterance audio
    utter, sr = librosa.core.load(path, hp.data.sr)
    # trim silent edges
    enrolled_utter, _ = librosa.effects.trim(utter, top_db=top_db)

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
        mel_spec, mfccs = cal_melspectrogram_mfcc(utter_part)

        if hp.model.input_size == 2:
            features = np.array([mel_spec, mfccs])
        else:
            features = np.array([mel_spec])

        utterances_spec.append(features)

        if stop:
            break

    return np.array(utterances_spec)


# https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
def cal_melspectrogram_mfcc(data):
    # calculate magnitude spectrogram with power 2, 
    # https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    # mel-scaled spectrogram of utterances, negative number, eg, -5.3575463
    mel_spec = librosa.feature.melspectrogram(
                                        y=data, 
                                        sr=hp.data.sr, 
                                        n_mels=hp.data.nmels, 
                                        n_fft=hp.data.nfft, 
                                        hop_length=int(hp.data.hop * hp.data.sr),
                                        win_length=int(hp.data.window * hp.data.sr),
                                    )

    # log-power mel spectrogram of utterances, negative number, eg, -5.3575463
    # shape=[num_mels, num_frames]
    log_power_mel_spec = librosa.power_to_db(mel_spec)

    # mfcc, shape=(n_mfcc, num_frames)
    mfccs = librosa.feature.mfcc(S=log_power_mel_spec, n_mfcc=hp.data.nmfccs)

    return log_power_mel_spec, mfccs

def save_spectrogram_tisv():
    """ 
    Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
    Each partial utterance is splitted by voice detection using DB and the first and the last 224 
    frames from each partial utterance are saved. 
    Need : utterance data set (TIMIT)
    """

    print("start text independent utterance feature extraction")
    assert hp.data.nmels == hp.data.nmfccs, 'nmels must be equal to nmfccs'

    # make folder to save train and test files
    if os.path.exists(hp.data.train_path):
        print('Folder [%s] exists, delete it ...' % hp.data.train_path)
        shutil.rmtree(hp.data.train_path)
    os.makedirs(hp.data.train_path)
    if os.path.exists(hp.data.test_path):
        print('Folder [%s] exists, delete it ...' % hp.data.test_path)
        shutil.rmtree(hp.data.test_path)
    os.makedirs(hp.data.test_path)

    # downloaded dataset path
    audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))
    total_speaker_num = len(audio_path)
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d, train speaker number: %d, test speaker number: %d" 
        % (total_speaker_num, train_speaker_num, total_speaker_num-train_speaker_num))

    # lower bound of utterance length
    sliding_window_length = int((hp.data.tisv_frame-1) * hp.data.hop * hp.data.sr)
    sliding_stride = int(sliding_window_length//2)
    win_length=int(hp.data.window * hp.data.sr)
    hop_length=int(hp.data.hop * hp.data.sr)
    print("utterrance sliding window length: %d, sliding stride: %d, mel window length: %d, mel hop length: %d" 
        % (sliding_window_length, sliding_stride, win_length, hop_length))

    min_train_utterances_num, min_test_utterances_num = 999, 999
    for i, folder in enumerate(audio_path):
        print("%d_th speaker under path [%s] processing ..." % (i, folder))
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.WAV':
                # path of each utterance
                utter_path = os.path.join(folder, utter_name)
                # load utterance audio
                utter, sr = librosa.core.load(utter_path, hp.data.sr)
                # trim silent edges
                utter, _ = librosa.effects.trim(utter, top_db=30)

                # padding iff the length is not enough but longer than sliding_window_length * 2 / 3
                if len(utter) < sliding_window_length:
                    if int(2*sliding_window_length//3) < len(utter):
                        print('\tutterance [%s], length: %d, does not have enough non-slience audio. Padding it ...' 
                            % (utter_name, len(utter)))
                        utter = librosa.util.pad_center(utter, sliding_window_length)
                    else:
                        print('\tutterance [%s], length: %d, is too short. skip it ...' 
                        % (utter_name, len(utter)))
                        continue

                stop = False
                for start in range(0, len(utter), sliding_stride):
                    end = start+sliding_window_length
                    if end >= len(utter):
                        start = len(utter)-sliding_window_length
                        end = len(utter)
                        stop = True

                    # cal mel spec and mfcc features
                    utter_part = utter[start:start+sliding_window_length]
                    mel_spec, mfccs = cal_melspectrogram_mfcc(utter_part)
                    
                    # normalize features
                    # norm_mel_spec = mel_spec/np.linalg.norm(mel_spec)
                    # norm_mfccs = mfccs/np.linalg.norm(mfccs)

                    if hp.model.input_size == 2:
                        features = np.array([mel_spec, mfccs])
                    else:
                        features = np.array([mel_spec])

                    utterances_spec.append(features)

                    if stop:
                        break

        utterances_spec = np.array(utterances_spec)

        if utterances_spec.shape[0] < hp.train.M:
            print('There are only %d utterances under path: %s, skip it' % (utterances_spec.shape[0], folder))
            continue

        # save spectrogram as numpy file 
        if i<train_speaker_num:
            min_train_utterances_num = min_train_utterances_num if utterances_spec.shape[0] > min_train_utterances_num else utterances_spec.shape[0]
            np.save(os.path.join(hp.data.train_path, "speaker%03d.npy"%i), utterances_spec)
        else:
            min_test_utterances_num = min_test_utterances_num if utterances_spec.shape[0] > min_test_utterances_num else utterances_spec.shape[0]
            np.save(os.path.join(hp.data.test_path, "speaker%03d.npy"%(i-train_speaker_num)), utterances_spec)

        print('\t%d partial utterances were extracted from original files ...' % utterances_spec.shape[0])

    print('min_train_utterances_num:', min_train_utterances_num)
    print('min_test_utterances_num:', min_test_utterances_num)


if __name__ == "__main__":
    save_spectrogram_tisv()








