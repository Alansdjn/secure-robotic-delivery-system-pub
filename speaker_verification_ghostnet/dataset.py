#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset

if __name__ == '__main__':
    from hparam import hparam as hp
elif __name__ == 'speaker_verification_ghostnet.dataset':
    from speaker_verification_ghostnet.hparam import hparam as hp

class TimitSpeakerDataset(Dataset):
    
    def __init__(self, training=None):
        # data path
        if training is None:
            if hp.training:
                self.path = hp.data.train_path
                self.utter_num = hp.train.M
            else:
                self.path = hp.data.test_path
                self.utter_num = hp.test.M
        elif training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M

        self.file_list = os.listdir(self.path)
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # load utterance spectrogram of selected speaker
        selected_file = random.sample(self.file_list, 1)[0]
        utters = np.load(os.path.join(self.path, selected_file))

        # utters shape: (m, 2, 112, 40), m depends on the speaker's utterance
        # select M utterances per speaker
        utter_index = np.random.randint(0, utters.shape[0], self.utter_num)

        # shape: (M, 2, 112, 40)
        return torch.tensor(utters[utter_index])







