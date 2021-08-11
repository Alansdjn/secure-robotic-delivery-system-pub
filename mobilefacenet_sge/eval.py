import os
from datetime import datetime
import time
import argparse
import scipy.io
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, SGD

from core.model import MobileFacenet
from core.margin import ArcMarginProduct
from core.utils import init_log, train, test, save_model
from core.dataset import CASIA_Face, LFW
from core.evaluator import evaluation_10_fold

def main(args):
    # init folder for current running
    save_dir = os.path.join(args.root, args.save_folder)
    result_folder = args.result_folder_prefix + str(args.sge) + datetime.now().strftime('_%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, result_folder)
    if os.path.exists(save_dir):
        raise NameError('model dir: [{}] exists!'.format(save_dir))
    os.makedirs(save_dir)

    # init log
    log = init_log(save_dir).info
    log('#'*100)
    log('Save dir: {}'.format(save_dir))
    log('Parameters: {}'.format(args))

    # init GPU
    assert torch.cuda.device_count() > 0
    multi_gpus = (torch.cuda.device_count() > 1)
    gpu_list = ','.join([str(id) for id in range(torch.cuda.device_count())])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    log('Avaliable GPU count: %d, GPU list: [%s]' % (torch.cuda.device_count(), gpu_list))

    # init dataloader
    worker_number = torch.get_num_threads()
    log('DataLoader worker number is: %d' % worker_number)

    # define testloader
    testset = LFW(os.path.join(args.root, args.test_dataset_folder))
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=worker_number)

    net = MobileFacenet().cuda().eval()
    net.load_state_dict(torch.load(args.mode_path)['net_state_dict'])

    if multi_gpus:
        net = nn.DataParallel(net)

    test(testloader, net, 0, save_dir, log)

    log('Finish training.')
    log('#'*100)
    log('\n\n\n')

if __name__ == '__main__':
    # process args
    parser = argparse.ArgumentParser(description='SGE-Train')
    parser.add_argument('--root', default='/data/acp20ww/secure-robotic-delivery-system/mobilefacenet_sge', 
        metavar='N', help='root path of the project')
    parser.add_argument('--mode_path', default='../model/best/sge64-mfn/best.ckpt', metavar='N', help='path of saved model')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', 
        help='batch size')
    parser.add_argument('--save_folder', default='model', metavar='N', 
        help='relative path to save model and tmp data')
    parser.add_argument('--result_folder_prefix', default='CASIA_112X96_EVAL_', metavar='N', 
        help='folder prefix, to identify which dataset is used to train this model')
    parser.add_argument('--test_dataset_folder', default='data/lfw', metavar='N', 
        help='relative path to test dataset')
    parser.add_argument('--sge', default=64, type=int, metavar='N', 
        help='Group number of class SpatialGroupEnhance')
    args = parser.parse_args()

    main(args)






