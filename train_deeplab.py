import os.path
import os.path as osp
import sys
from collections import deque

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from options.deeplab_options import BaseOptions
from torch.utils.tensorboard import SummaryWriter

from models.deeplab import Deeplab
import data

def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss2d(weight=weights, size_average=True,
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score), label)
    return loss

# parse options
opt = BaseOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

net = Deeplab(num_classes=opt.label_nc, init_weights='DeepLab_init.pth', restore_from=None, phase='train')
net.cuda()
transform = []
target_transform = []

optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.99,
                        weight_decay=0.0005)
#scheduler= torch.optim.lr_schedulre.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
writer = SummaryWriter(log_dir=opt.checkpoints_dir)

iteration = 0
losses = deque(maxlen=10)
while True:
    for i, data_i in enumerate(dataloader):
        # Clear out gradients
        optimizer.zero_grad()
        net.adjust_learning_rate(opt.lr, optimizer, iteration)

        # load data/label
        #im = make_variable(im, requires_grad=False)
        #label = make_variable(label, requires_grad=False)

        # forward pass and compute loss
        im = data_i['image_seg'].cuda()
        label = data_i['label'].cuda().long().squeeze(1)
        preds = net(im)
        loss = torch.nn.CrossEntropyLoss(ignore_index=opt.label_nc)(preds, label)

        # backward pass
        loss.backward()
        writer.add_scalar('Loss/train', loss.item(), iteration)
        writer.add_scalar('Lr/train', optimizer.param_groups[0]['lr'], iteration)
        losses.append(loss.item())

        # step gradients
        optimizer.step()

        # log results
        if iteration % 10 == 0:
            print('Iteration {}:\t{}'.format(iteration, np.mean(losses)))
        iteration += 1

        if iteration % opt.snapshot == 0:
            torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, '{}-iter{}.pth'.format(opt.name, iteration)))
        if iteration >= opt.niter:
            print('Optimization complete.')
            break
    if iteration >= opt.niter:
        break

    #scheduler.step()
