#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 23:51:05 2023

@author: senlin
"""
import os
import time
import math
from numbers import Number
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import lib.dist as dist
import lib.utils as utils
from lib.flows import FactorialNormalizingFlow

from elbo_decomposition import elbo_decomposition
from CNNlayer import ConvEncoder
from CNNlayer import ConvDecoder
import VAE
from scipy.io import loadmat

Kol=loadmat('Kol_Label.mat')
w=Kol['w'] #This data is separated every 5 time units
w = torch.FloatTensor(w)

# split traning and test set 5:5, and standardize them
train_set = w[:w.size()[0]//2,:,:]
trainset_scaled = (train_set-train_set.mean())/train_set.std()
test_set = w[w.size()[0]//2:,:,:]
testset_scaled = (test_set-test_set.mean())/test_set.std()

train_loader = VAE.setup_data_loaders(trainset_scaled, use_cuda=True) 
prior_dist = dist.Normal()
q_dist = dist.Normal()


vae = VAE.VAE(z_dim=4, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
    include_mutinfo=True, tcvae=True, conv=True, mss=False)

# setup the optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)
train_elbo = []

# training loop
dataset_size = len(train_loader.dataset)
num_iterations = len(train_loader) * 400 #num of epochs: 1
iteration = 0
# initialize loss accumulator
elbo_running_mean = utils.RunningAverageMeter()
while iteration < num_iterations:
    for i, x in enumerate(train_loader):
        iteration += 1
        batch_time = time.time()
        vae.train()
        #anneal_kl(args, vae, iteration)
        vae.lamb = 0
        vae.beta = 10
        optimizer.zero_grad()
        # transfer to GPU
        x = x.cuda(non_blocking=True)
        # wrap the mini-batch in a PyTorch Variable
        x = Variable(x)
        # do ELBO gradient and accumulate loss
        obj, elbo,reconst_error = vae.elbo(x, dataset_size)
        if utils.isnan(obj).any():
            raise ValueError('NaN spotted in objective.')
        obj.mean().mul(-1).backward()
        elbo_running_mean.update(elbo.mean().item())
        optimizer.step()

        # report training diagnostics
        if iteration % 5 == 0:  #args.log_freq: 200
            train_elbo.append(elbo_running_mean.avg)
            print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f) reconst: %.2f' % (
                iteration, time.time() - batch_time, vae.beta, vae.lamb,
                elbo_running_mean.val, elbo_running_mean.avg, reconst_error.mean()))

            vae.eval()
            
vae.eval()
utils.save_checkpoint({
    'state_dict': vae.state_dict(),
    #'args': args
}, 'test1', 0)

testset_scaled = testset_scaled.cuda()
xs,x_param,zs,z_test = vae.reconstruct_img(testset_scaled)
xs,x_param,zs,z_train = vae.reconstruct_img(trainset_scaled)
VAElatent = torch.cat((z_train,z_test),1)
np.save('allVAElatent',VAElatent.cpu().detach().numpy())

