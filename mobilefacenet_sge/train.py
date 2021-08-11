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

    # define trainloader
    train_root = os.path.join(args.root, args.train_dataset_folder)
    trainset = CASIA_Face(root=train_root, downsample_rate=args.sample_rate)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=worker_number)
    
    # define testloader
    test_root = os.path.join(args.root, args.test_dataset_folder)
    testset = LFW(test_root)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=worker_number)

    # define model
    net = MobileFacenet(sgegroup=args.sge).cuda()
    arc_margin = ArcMarginProduct(128, trainset.class_nums).cuda()

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # define optimizers
    ignored_params = list(map(id, net.linear1x1.parameters()))
    ignored_params += list(map(id, arc_margin.weight))
    prelu_params_id = []
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1x1.parameters(), 'weight_decay': 4e-4},
        {'params': arc_margin.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    if multi_gpus:
        net = nn.DataParallel(net)
        arc_margin = nn.DataParallel(arc_margin)

    for epoch in range(start_epoch, args.epochs+1):
        curr_lr = ', '.join([str(lr) for lr in exp_lr_scheduler.get_last_lr()])

        # train model
        log('Train Epoch: [{0:>2}] / [{1:>2}], lr: [{2}] ...'\
            .format(epoch, args.epochs, curr_lr))
        train(trainloader, net, arc_margin, loss_fn, optimizer, epoch, log)

        # test model on lfw
        log('Test Epoch: [{0:>2}] ...'.format(epoch))
        test(testloader, net, epoch, save_dir, log)

        # save model
        log('Saving checkpoint: [{0:>2}]'.format(epoch))
        save_model(save_dir, net, epoch, multi_gpus)

        exp_lr_scheduler.step()

    log('Finish training.')
    log('#'*100)
    log('\n\n\n')

if __name__ == '__main__':
    # process args
    parser = argparse.ArgumentParser(description='SGE-Train')
    parser.add_argument('--root', default='/data/acp20ww/secure-robotic-delivery-system/mobilefacenet_sge', 
        metavar='N', help='root path of the project')
    parser.add_argument('--resume', default='', metavar='N', help='path of saved model')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', 
        help='batch size')
    parser.add_argument('--save_folder', default='model', metavar='N', 
        help='relative path to save model and tmp data')
    parser.add_argument('--result_folder_prefix', default='CASIA_112X96_', metavar='N', 
        help='folder prefix, to identify which dataset is used to train this model')
    parser.add_argument('--train_dataset_folder', default='data/CASIA', metavar='N', 
        help='relative path to training dataset')
    parser.add_argument('--test_dataset_folder', default='data/lfw', metavar='N', 
        help='relative path to test dataset')
    parser.add_argument('--sge', default=64, type=int, metavar='N', 
        help='Group number of class SpatialGroupEnhance')
    parser.add_argument('--sample_rate', default=0.4, type=float, metavar='N', 
        help='Use a smaller dataset or subset of Web Face')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='Total epochs')
    parser.add_argument('--milestones', default=[25, 37, 43], nargs='+', type=int, 
        metavar='N', help='milestones')
    args = parser.parse_args()

    main(args)

