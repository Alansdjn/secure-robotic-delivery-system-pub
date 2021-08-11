import __main__
import os
import logging
from pathlib import Path
import torch
import random
import numpy as np
from datetime import datetime
import time
import argparse
import scipy.io
import shutil

if __name__ == '__main__':
    from evaluator import evaluation_10_fold
elif __name__ == 'core.utils':
    from core.evaluator import evaluation_10_fold
elif __name__ == 'mobilefacenet_sge.core.utils':
    from mobilefacenet_sge.core.evaluator import evaluation_10_fold

__all__ = ['init_log', 'currtime', 'minitues', 'seconds', 'train', 'test', 'save_model']

def init_log(output_dir):
    msg_fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_fmt='%m/%d/%Y %H:%M:%S'
    log_filename = os.path.join(output_dir, 'log.log')
    logger_name = Path(__main__.__file__).stem

    logging.basicConfig(level=logging.DEBUG, format=msg_fmt, datefmt=date_fmt, 
                        filename=log_filename, filemode='a')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(fmt=msg_fmt, datefmt=date_fmt)

    # add formatter to ch
    ch.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # add ch to logger
    logger.addHandler(ch)

    return logger

def currtime():
    return time.ctime()[11:19]

def minitues(time_elapsed):
    return int(time_elapsed // 60)

def seconds(time_elapsed):
    return int(time_elapsed % 60)

def train(dataloader, model, margin, loss_fn, optimizer, epoch, logger):
    # train mode
    model.train()

    total_batch = len(dataloader)
    train_loss = 0.0
    since = time.time()
    for i, data in enumerate(dataloader):
        img, label = data[0].cuda(), data[1].cuda()

        # Forward pass: compute predicted y by passing x to the model.
        output = margin(model(img), label)

        # Compute loss.
        batch_size = img.size(0)
        batch_loss = loss_fn(output, label)
        train_loss += batch_loss.item() * batch_size

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters     
        batch_loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        # print start current step
        if i%200==0:
            logger('    [Epoch:{:>2}] Processed: [{:>3}] / [{:>3}] ...'\
                .format(epoch, i+1, total_batch))

    # print current epoch loss and elapsed time
    train_avg_loss = train_loss / len(dataloader.dataset)
    time_elapsed = time.time() - since
    logger('    [Epoch:{:>2}] Loss: {:.4f}, Elapsed time: {}m {}s'\
        .format(epoch, train_avg_loss, minitues(time_elapsed), seconds(time_elapsed)))

    return train_avg_loss

def test(dataloader, model, epoch, root, logger):
    # eval mode
    model.eval()

    total_batch = len(dataloader)
    featureLs = None
    featureRs = None
    for i, data in enumerate(dataloader):
        res = [model(d.cuda()).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)

        featureLs = featureL if featureLs is None else np.concatenate((featureLs, featureL), 0)
        featureRs = featureR if featureRs is None else np.concatenate((featureRs, featureR), 0)

        # print start current step
        if i%50==0:
            logger('    [Epoch:{:>2}] Processed: [{:>3}] / [{:>3}] ...'\
                .format(epoch, i+1, total_batch))

    # save tmp_result
    folds = dataloader.dataset.folds
    flags = dataloader.dataset.flags
    result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
    tmp_result_path = os.path.join(root, 'tmp_result.mat')
    scipy.io.savemat(tmp_result_path, result)
    acc = np.mean(evaluation_10_fold(tmp_result_path))
    logger('    [Epoch:{:>2}] Accuracy: {:.4f}'.format(epoch, acc * 100))

    return acc

def save_model(root, model, epoch, multi_gpu=True):
    if not os.path.exists(root):
        os.mkdir(root)
    
    if multi_gpu:
        torch.save(
            {'epoch': epoch, 'net_state_dict': model.module.state_dict()}, 
            os.path.join(root, '%03d.ckpt' % epoch)
        )
    else:
        torch.save(
            {'epoch': epoch, 'net_state_dict': model.state_dict()}, 
            os.path.join(root, '%03d.ckpt' % epoch)
        )

def copy_best_model(source, destination, logger):
    # cp best model to model/best/cnn{?}/best.ckpt
    shutil.copyfile(source, destination)
    try:
        shutil.copyfile(source, destination)
        logger("File copied successfully {} -> {}.".format(source, destination))
    # If source and destination are same
    except shutil.SameFileError:
        logger("Source and destination represents the same file.")
    # If destination is a directory.
    except IsADirectoryError:
        logger("Destination is a directory.")
    # If there is any permission issue
    except PermissionError:
        logger("Permission denied.")
    # For other errors
    except:
        logger("Error occurred while copying file.")

if __name__ == '__main__':
    # logger = init_log('./')
    # logger.info('aaa')
    pass






