import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

class SpokenDigitDataset(Dataset):
    def __init__(self, file=None, rootdir=None, transform=None):
        self.df = pd.read_csv(file)
        self.rootdir = rootdir # root directory for the images
        self.transform = transform
        self.class_num = self.df['Label'].nunique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        fname, label = row['File'], row['Label']
        path = os.path.join(self.rootdir, fname[:-4] + '.jpg')
        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        
        # return image tensor and numeric label
        return img, torch.tensor(DIGITS.index(label)) 
