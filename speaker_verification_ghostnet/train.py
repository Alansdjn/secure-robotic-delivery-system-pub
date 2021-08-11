#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hparam import hparam as hp
from dataset import TimitSpeakerDataset
from ge2e_loss import GE2ELoss
from ghostnet import ghostnet
from utils import get_centroids, get_cossim


def format_time():
    return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())


def print_msg(epoch, batch, total_batch, iteration, loss, loss_type='batch'):
    """ 
    loss_type: 
        batch, current batch loss
        epoch, average epoch loss

    """
    if loss_type == 'batch':
        msg_template = "{} - Epoch: [{:>4}:{:>4}], Batch: [{:>4}/{:>4}], Iteration: {:>6}, Batch loss: {:.4f}"
        mesg = msg_template.format(format_time(), epoch, hp.train.epochs, batch, total_batch, iteration, loss)
    elif loss_type == 'epoch':
        msg_template = "{} - Epoch: [{:>4}:{:>4}], Iteration: {:>6}, Avg loss: {:.4f}"
        mesg = msg_template.format(format_time(), epoch, hp.train.epochs, iteration, loss / batch)
    print(mesg)


def save_final_model(embedder_net, ge2e_loss, optimizer, scheduler, epoch, iteration):
    timestr=time.strftime('%Y%m%d%H%M%S', time.localtime())
    
    embedder_net.eval().cpu()
    ge2e_loss.eval().cpu()
    save_model_filename = "{}_embeddernet_ge2eloss_e{:04d}_t{}.model".format(hp.uid, epoch, timestr)
    current_train_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)

    torch.save({
        'embedder_net_dict': embedder_net.state_dict(),
        'ge2e_loss_dict': ge2e_loss.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'uid': hp.uid,
        }, current_train_path)

    torch.save({
        'embedder_net_dict': embedder_net.state_dict(),
        'ge2e_loss_dict': ge2e_loss.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'uid': hp.uid,
        }, hp.model.model_path)
    print("Save model: %s and %s" % (current_train_path, hp.model.model_path))


def restore_model(embedder_net, ge2e_loss, optimizer, scheduler):
    if not os.path.isfile(hp.model.model_path):
        print('File does not exist, can not restore model from: %s' % hp.model.model_path)
        return None

    print('Restore model from: %s' % hp.model.model_path)
    load_model = torch.load(hp.model.model_path)
    embedder_net.load_state_dict(load_model['embedder_net_dict'])
    ge2e_loss.load_state_dict(load_model['ge2e_loss_dict'])
    optimizer.load_state_dict(load_model['optimizer_dict'])
    scheduler.load_state_dict(load_model['scheduler_dict'])
    start = load_model['epoch']
    iteration = load_model['iteration']
    uid = load_model['uid']

    return embedder_net, ge2e_loss, optimizer, scheduler, start, iteration, uid


# decrease
def embedder_net_lr(epoch):
    for power, milestone in enumerate(hp.train.milestones):
        if epoch < milestone:
            return hp.train.gamma**power

    return hp.train.gamma**len(hp.train.milestones)


def ge2e_loss_lr(epoch):
    # return 1
    for power, milestone in enumerate(hp.train.milestones):
        if epoch < milestone:
            return hp.train.ge2e_gamma**power

    return hp.train.ge2e_gamma**len(hp.train.milestones)


def train():
    print('='*120)
    print('Start training ...')
    print('='*120)
    print('parameters:', hp)

    # init multi gpu
    gpu_list = None
    if torch.cuda.device_count() > 1:
        gpu_list = ','.join([str(id) for id in range(torch.cuda.device_count())])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)

    worker_numbers = torch.get_num_threads()
    print('DataLoader worker number is: %d' % worker_numbers)
    
    train_dataset = TimitSpeakerDataset()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=worker_numbers, drop_last=True) 

    test_dataset = TimitSpeakerDataset(training=False)
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=worker_numbers, drop_last=True)
    
    device = torch.device(hp.device)
    embedder_net = ghostnet(inp=hp.model.input_size, num_classes=hp.model.feature_size).to(device)
    ge2e_loss = GE2ELoss(device)

    #Both net and loss have trainable parameters
    optimizer = torch.optim.SGD(
                [
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], 
                lr=hp.train.lr, 
                momentum=0.9, 
                weight_decay=5e-4
            )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=[embedder_net_lr, ge2e_loss_lr]
        )

    start = 0
    iteration = 0
    if hp.train.restore:
        restored_model = restore_model(embedder_net, ge2e_loss, optimizer, scheduler)
        if restored_model is not None:
            embedder_net, ge2e_loss, optimizer, scheduler, start, iteration, uid = restored_model
            hp.uid = uid if uid is not None else hp.uid
    
    real_batch_size = hp.train.N*hp.train.M
    hp.train.epochs += start
    for e in range(start, hp.train.epochs):
        print('='*120)
        embedder_net.train()

        lrs = ', '.join([str(lr) for lr in scheduler.get_last_lr()])
        print("{} - Epoch: [{:>4}:{:>4}], Learning rate: [{}]".format(format_time(), e+1, hp.train.epochs, lrs))

        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader): 
            mel_db_batch = mel_db_batch.to(device)
            # shape: (N, M, 2, frames, nmels) -> (N * M, 2, frames, nmels)
            mel_db_batch = torch.reshape(mel_db_batch, 
                (real_batch_size, mel_db_batch.size(2), mel_db_batch.size(3), mel_db_batch.size(4)))
            perm = random.sample(range(0, real_batch_size), real_batch_size)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            
            # shape: (N * M, 2, frames, nmels)
            embeddings = embedder_net(mel_db_batch)
            # shape: (N * M, hp.model.proj, 1)
            embeddings = embeddings[unperm]

            # get loss, call backward, step optimizer
            # wants (Speaker, Utterances, embedding_size)
            # shape: (N * M, hp.model.proj, 1) -> (N, M, hp.model.proj)
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            loss = ge2e_loss(embeddings) 

            optimizer.zero_grad()

            loss.backward()
            # # avoid vanishing gradients
            # torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            # torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)

            optimizer.step()
            
            total_loss += loss
            iteration += 1

            # print batch loss
            if (batch_id+1) % hp.train.log_interval == 0:
                print_msg(e+1, batch_id+1, len(train_loader), iteration, loss, loss_type='batch')
        
        # print current epoch avg loss
        print_msg(e+1, batch_id+1, len(train_loader), iteration, total_loss, loss_type='epoch')

        # cal test EER
        if epoch % hp.train.eer_interval == 0:
            avg_EER = cal_eer(embedder_net, test_loader, device, print_detail=False)
            msg_template = "{} - Epoch: [{:>4}:{:>4}], Iteration: {:>6}, Avg EER ({} epochs): {:.4f}"
            mesg = msg_template.format(format_time(), epoch+1, hp.train.epochs, iteration, hp.test.epochs, avg_EER)
            print(mesg)
        
        scheduler.step()

    #save model
    print("="*120)
    save_final_model(embedder_net, ge2e_loss, optimizer, scheduler, hp.train.epochs, iteration)

    print("="*120)
    print("Training done")
    print("="*120)


def find_best_eer(sim_matrix):
    # calculating EER
    diff=1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0

    for thres in [0.01*i+0.5 for i in range(50)]:
        sim_matrix_thresh = sim_matrix>thres
        
        FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
        /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)

        FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
        /(float(hp.test.M/2))/hp.test.N)
        
        # Save threshold when FAR = FRR (=EER)
        # False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER)
        if diff> abs(FAR-FRR):
            diff = abs(FAR-FRR)
            EER = (FAR+FRR)/2
            EER_thresh = thres
            EER_FAR = FAR
            EER_FRR = FRR

    return EER, EER_thresh, EER_FAR, EER_FRR

def cal_eer(embedder_net, test_loader, device, print_detail=False):
    assert hp.test.M % 2 == 0, 'hp.test.M % 2 == 0'
    embedder_net.eval()
    
    epoch_total_EER = 0
    half_batch_size = hp.test.N*hp.test.M//2
    for e in range(hp.test.epochs):
        if print_detail: 
            print('='*120)

        batch_total_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            mel_db_batch = mel_db_batch.to(device)
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            
            enrollment_batch = torch.reshape(enrollment_batch, 
                (half_batch_size, enrollment_batch.size(2), enrollment_batch.size(3), enrollment_batch.size(4)))
            verification_batch = torch.reshape(verification_batch, 
                (half_batch_size, verification_batch.size(2), verification_batch.size(3), verification_batch.size(4)))
            
            perm = random.sample(range(0, half_batch_size), half_batch_size)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, 
                (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, 
                (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            
            EER, EER_thresh, EER_FAR, EER_FRR = find_best_eer(sim_matrix)
            batch_total_EER += EER

            if print_detail:
                msg_template = "{} - Epoch: [{:>4}:{:>4}], Iteration: {:>6}, EER: {:0.2f}, Threshold: {:0.2f}, FAR: {:0.2f}, FRR: {:0.2f}"
                mesg = msg_template.format(format_time(), e+1, hp.test.epochs, batch_id, EER, EER_thresh, EER_FAR, EER_FRR)
                print(mesg)            
        
        avg_batch_EER = batch_total_EER/(batch_id+1)
        epoch_total_EER += avg_batch_EER

        if print_detail: 
            print("{} - Epoch: [{:>4}:{:>4}], Avg EER: {:0.2f}".format(format_time(), e+1, hp.test.epochs, avg_batch_EER))

    avg_EER = epoch_total_EER / hp.test.epochs
    return avg_EER


def test():
    print('='*120)
    print('Start tesing ...')
    print('='*120)
    print('parameters:', hp)

    worker_numbers = torch.get_num_threads()
    print('DataLoader worker number is: %d' % worker_numbers)

    test_dataset = TimitSpeakerDataset()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=worker_numbers, drop_last=True)
    
    print('Load model from: %s' % hp.model.model_path)
    device = torch.device('cpu')
    embedder_net = ghostnet(inp=hp.model.input_size, num_classes=hp.model.feature_size).to(device)
    embedder_net.load_state_dict(torch.load(hp.model.model_path)['embedder_net_dict'])
    embedder_net.eval()

    # cal test EER
    avg_EER = cal_eer(embedder_net, test_loader, device, print_detail=True)

    print("="*120)
    print("EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
    print("="*120)
    print("Testing done")
    print("="*120)
        
if __name__=="__main__":
    if hp.training:
        train()
        hp.training = False
        test()
    else:
        test()
