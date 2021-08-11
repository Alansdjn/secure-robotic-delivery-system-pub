import __main__
import os
import logging
from pathlib import Path
import torch
import numpy as np
import time
import shutil

if __name__ == '__main__':
    from model import FeatureExtractor, SpokenDigitModel
elif __name__ == 'core.utils':
    from core.model import FeatureExtractor, SpokenDigitModel
elif __name__ == 'cnn_xgboost.core.utils':
    from cnn_xgboost.core.model import FeatureExtractor, SpokenDigitModel

__all__ = ['init_log', 'currtime', 'minitues', 'seconds', 'train', 'test', 'save_model', 'getFeatures']

def init_log(output_dir):
    msg_fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_fmt='%m/%d/%Y %H:%M:%S'
    log_filename = os.path.join(output_dir, 'log.log')
    logger_name = Path(__main__.__file__).stem

    logging.basicConfig(
        level=logging.DEBUG, 
        format=msg_fmt, 
        datefmt=date_fmt, 
        filename=log_filename, 
        filemode='a'
        )

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

def train(dataloader, model, loss_fn, optimizer, epoch, logger):
    # train mode
    model.train()

    total_batch = len(dataloader)
    train_loss = 0.0
    since = time.time()
    for i, data in enumerate(dataloader):
        img, label = data[0].cuda(), data[1].cuda()

        # Forward pass: compute predicted y by passing x to the model.
        output = model(img)

        # Compute loss.
        batch_loss = loss_fn(output, label)
        batch_size = img.size(0)
        train_loss += batch_loss.item() * batch_size

        # Before the backward pass, use the optimizer object to zero all of the gradients for the 
        # variables it will update (which are the learnable weights of the model). This is because 
        # by default, gradients are accumulated in buffers( i.e, not overwritten) whenever 
        # .backward() is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters     
        batch_loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        # print start current step
        if i%50==0:
            logger('    [Epoch:{:>2}] Processed: [{:>3}] / [{:>3}] ...'\
                .format(epoch, i+1, total_batch))
    
    # print current epoch loss and elapsed time
    train_avg_loss = train_loss / len(dataloader.dataset)
    time_elapsed = time.time() - since
    logger('    [Epoch:{:>2}] Loss: {:.4f}, Elapsed time: {}m {}s'\
        .format(epoch, train_avg_loss, minitues(time_elapsed), seconds(time_elapsed)))

    return train_avg_loss

def test(dataloader, model, loss_fn, epoch, logger):
    # eval mode
    model.eval()
    
    total_batch = len(dataloader)
    test_loss, correct = 0, 0
    for i, data in enumerate(dataloader):
        img, label = data[0].cuda(), data[1].cuda()

        # Forward pass: compute predicted y by passing x to the model.
        outputs = model(img)

        # Compute loss.
        batch_loss = loss_fn(outputs, label)
        batch_size = img.size(0)
        test_loss += batch_loss.item() * batch_size

        # Compute correct prediction count
        correct += (outputs.argmax(1) == label).type(torch.float).sum().item()

        # print start current step
        if i%30==0:
            logger('    [Epoch:{:>2}] Processed: [{:>3}] / [{:>3}] ...'\
                .format(epoch, i+1, total_batch))

    size = len(dataloader.dataset)
    test_avg_loss = test_loss / size
    accuracy = correct / size
    logger('    [Epoch:{:>2}] Accuracy: {:.4f}%, Avg loss:{:>.4f}'\
        .format(epoch, (100*accuracy), test_avg_loss))

    return test_avg_loss, accuracy

def save_model(root, model, epoch, multi_gpus=True):
    if not os.path.exists(root):
        os.mkdir(root)

    if multi_gpus:
        torch.save(
            {'epoch': epoch, 'net_state_dict': model.module.state_dict()}, 
            os.path.join(root, '%03d.ckpt' % epoch)
            )
    else:
        torch.save(
            {'epoch': epoch, 'net_state_dict': model.state_dict()}, 
            os.path.join(root, '%03d.ckpt' % epoch)
            )


def getFeatures(resume, input_size, trainloader, testloader):
    # load tmp 
    # if os.path.isfile('../Output/features_from_ghostnet.npy'):
    #     print('tmp file exist, loading ...')
    #     with open('../Output/features_from_ghostnet.npy', 'rb') as f:
    #         train_features = np.load(f)
    #         train_labels = np.load(f)
    #         test_features = np.load(f)
    #         test_labels = np.load(f)
    #     return train_features, train_labels, test_features, test_labels

    # load pre-trained ghostnet model
    ckpt = torch.load(resume, map_location=torch.device('cpu'))
    net = SpokenDigitModel(input_size=input_size).eval()
    net.load_state_dict(ckpt['net_state_dict'])
    feature_extractor = FeatureExtractor(net)

    train_features = []
    train_labels = []
    for i, (inputs, labels) in enumerate(trainloader):
        train_features += feature_extractor(inputs)
        train_labels += labels.tolist()

        if i % 30 ==0:
            print(currtime(), i, len(trainloader))

    test_features = []
    test_labels = []
    for i, (inputs, labels) in enumerate(testloader):
        test_features += feature_extractor(inputs)
        test_labels += labels.tolist()

        if i % 10 ==0:
            print(currtime(), i, len(testloader))

    # save tmp files
    # with open('../Output/features_from_ghostnet.npy', 'wb') as f:
    #     print('tmp file does not exist, save data, loading ...')
    #     np.save(f, np.array(train_features))
    #     np.save(f, np.array(train_labels))
    #     np.save(f, np.array(test_features))
    #     np.save(f, np.array(test_labels))
    
    return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels)

def copy_best_model(source, destination, logger):
    # cp best model to model/best/cnn{?}/best.ckpt
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
    source = 'plot.py'
    destination = '../model/best/cnn3/test.py'

    copy_best_model(source, destination, print)






