import os
import numpy as np
import random
from PIL import Image

import torch
from torch.utils.data import Dataset

class CASIA_Face(Dataset):
    def __init__(self, root, downsample_rate=1.0):
        with open(os.path.join(root, 'CASIA-WebFace-112X96.txt')) as f:
            img_label_list = f.read().splitlines()

        image_list, label_list = [], []
        # only use first sample_count elements
        sample_count = int(len(img_label_list) * downsample_rate)
        for i, info in enumerate(img_label_list):
            if i >= sample_count:
                break

            image_dir, label_name = info.split(' ')
            image_list.append(os.path.join(root, 'CASIA-WebFace-112X96', image_dir))
            label_list.append(label_name)

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = int(self.label_list[index])

        img = np.array(Image.open(img_path))
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        flip = np.random.choice([-1, 1])
        if flip == -1:
            img = img[:, ::flip, :]

        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)

class LFW(Dataset):
    def __init__(self, root):
        nameLs, nameRs, folds, flags = LFW.parse(root)
        self.imgl_list = nameLs
        self.imgr_list = nameRs
        self.folds = folds
        self.flags = flags

    def __getitem__(self, index):
        imgl = np.array(Image.open(self.imgl_list[index]))
        imgr = np.array(Image.open(self.imgr_list[index]))
        return LFW.getitem(imgl, imgr)

    def __len__(self):
        return len(self.imgl_list)

    @staticmethod
    def parse(root):
        with open(os.path.join(root, 'pairs.txt')) as f:
            pairs = f.read().splitlines()[1:]

        folder_name = 'lfw-112X96'
        nameLs, nameRs, folds, flags = [], [], [], []
        for i, p in enumerate(pairs):
            fold = i // 600

            p = p.split('\t')
            if len(p) == 3:
                nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
                nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
                flag = 1
            elif len(p) == 4:
                nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
                nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
                flag = -1

            nameLs.append(nameL)
            nameRs.append(nameR)
            folds.append(fold)
            flags.append(flag)

        return [nameLs, nameRs, folds, flags]

    @staticmethod
    def getitem(imgl, imgr):
        """ imgl: array, imgr: array """
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)

        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        # imgl = imgl[:, :, ::-1]
        # imgr = imgr[:, :, ::-1]
        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]
        return imgs


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    testset = LFW('../data/lfw/')
    testloader = DataLoader(testset, batch_size=32, num_workers=1)
    for batch in testloader:
        print(batch[0].shape)
        break






