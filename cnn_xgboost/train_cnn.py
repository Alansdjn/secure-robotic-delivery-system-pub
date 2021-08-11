import os
from datetime import datetime
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.optim import lr_scheduler, SGD, Adam
import torchvision.transforms as transforms

from core.utils import init_log, train, test, save_model, copy_best_model
from core.dataset import SpokenDigitDataset
from core.model import SpokenDigitModel, FeatureExtractor
from core.plot import lineplot

def main(args):
    # init result folder
    result_root = os.path.join(args.root, args.save_folder)
    result_folder = args.result_folder_prefix + str(args.input_size) + datetime.now().strftime('_%Y%m%d_%H%M%S')
    save_dir = os.path.join(result_root, result_folder)
    if os.path.exists(save_dir):
        raise NameError('model dir: [{}] exists!'.format(save_dir))
    os.makedirs(save_dir)

    # init log
    log = init_log(save_dir).info
    log('#'*100)
    log(save_dir)
    log('Parameters: {}'.format(args))

    # init GPU
    assert torch.cuda.device_count() > 0
    multi_gpus = (torch.cuda.device_count() > 1)
    gpu_list = ','.join([str(id) for id in range(torch.cuda.device_count())])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    log('Avaliable GPU count: {}, GPU list: {}'.format(torch.cuda.device_count(), gpu_list))

    # init dataloader
    # define trainloader and testloader
    dataset_file = os.path.join(args.root, args.dataset_folder, 'spoken_digit.csv')
    image_root = os.path.join(args.root, args.image_folder)
    meldset = SpokenDigitDataset(dataset_file, image_root, transforms.ToTensor())

    size = len(meldset)
    val_size = int(0.2 * size)
    train_size = size - val_size
    generator=torch.Generator().manual_seed(args.seed)
    train_dset, val_dset = random_split(meldset, [train_size, val_size], generator=generator)

    worker_number = torch.get_num_threads()
    log('DataLoader worker number is: %d' % worker_number)

    trainloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=worker_number)
    testloader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=worker_number)
    log("Number of samples in train set: {}".format(train_size))
    log("Number of samples in validation set: {}".format(val_size))

    # define model
    net = SpokenDigitModel(input_size=args.input_size).cuda()

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    learning_rate = 0.001
    # optimizer = SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=4e-5, nesterov=True)
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=4e-5)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.3)
    loss_fn = nn.CrossEntropyLoss()

    if multi_gpus:
        net = nn.DataParallel(net)

    train_losses = []
    test_losses = []
    accuracies = []
    for epoch in range(start_epoch, args.epochs+1):
        curr_lr = ', '.join([str(lr) for lr in exp_lr_scheduler.get_last_lr()])

        # train model
        log('Train Epoch: [{0:>2}] / [{1:>2}], lr: [{2}] ...'\
            .format(epoch, args.epochs, curr_lr))
        train_loss = train(trainloader, net, loss_fn, optimizer, epoch, log)
        train_losses.append(train_loss)

        # test model on lfw
        log('Test Epoch: [{0:>2}] ...'.format(epoch))
        test_loss, accuracy = test(testloader, net, loss_fn, epoch, log)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        # save model
        log('Saving checkpoint: [{0:>2}] ...'.format(epoch))
        save_model(save_dir, net, epoch, multi_gpus)

        exp_lr_scheduler.step()

    # copy save model with highest accuracy to best folder
    source = '{}/{:03d}.ckpt'.format(save_dir, np.argmin(test_losses))
    destination = './model/best/cnn{}/best.ckpt'.format(args.input_size)
    copy_best_model(source, destination, log)

    # plot loss 
    y = [train_losses, test_losses]
    x = [list(range(1, len(i)+1)) for i in y]
    labels = ['train_loss', 'test_loss']
    lineplot(x, y, args.milestones, labels, './data/cnn_loss.png')

    log('Finish training.')
    log('#'*100)
    log('\n\n\n')

if __name__ == '__main__':
    import argparse

    # process args
    parser = argparse.ArgumentParser(description='CNN-Train')
    parser.add_argument('--root', 
        default='/data/acp20ww/secure-robotic-delivery-system/', 
        metavar='N', help='root path of the project')
    parser.add_argument('--resume', default='', metavar='N', help='path of saved model')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='batch size')
    parser.add_argument('--save_folder', default='cnn_xgboost/model', metavar='N', 
        help='relative path to save model and tmp data')
    parser.add_argument('--result_folder_prefix', default='SPOKEN_DIGIT_', metavar='N', 
        help='folder prefix, to identify which dataset is used to train this model')
    parser.add_argument('--dataset_folder', default='ghostnet/data/speech_commands_v0.01', 
        metavar='N', help='relative path to training dataset config file')
    parser.add_argument('--image_folder', default='cnn_xgboost/data/melimages_55x55_png', 
        metavar='N', help='relative path to training image')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='Total epochs')
    parser.add_argument('--milestones', default=[25, 37, 43], nargs='+', type=int, metavar='N', 
        help='Total epochs')
    parser.add_argument('--seed', default=200207124, type=int, metavar='N', help='seed')
    parser.add_argument('--input_size', default=3, type=int, metavar='N', help='input channels')
    args = parser.parse_args()

    main(args)





