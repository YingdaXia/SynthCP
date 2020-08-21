import os.path
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collections import deque
from tqdm import *

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from util.util import tensor2im
from util.eval_util import fast_hist, result_stats
from options.deeplab_options import BaseOptions

from models.deeplab import Deeplab
import data
import json
import pdb

ignore_label = 255
id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

# parse options
opt = BaseOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

net = Deeplab(num_classes=opt.label_nc, init_weights=None, restore_from=None, phase='train')
net.load_state_dict(torch.load(opt.model_path, map_location='cuda:{}'.format(opt.gpu_ids[0])))
net.cuda()
net.eval()
net.set_dropout_train_mode()

hist = np.zeros((19, 19))
metrics = []
samples = 50
iterations = tqdm(enumerate(dataloader), total=len(dataloader))
for i, data_i in iterations:
    # Clear out gradients

    # forward pass and compute loss
    im = data_i['image_seg'].cuda()
    label = data_i['label'].squeeze(1)

    outputs = torch.zeros(
                    samples,
                    im.shape[0],
                    opt.label_nc,
                    im.shape[2],
                    im.shape[3],
                ).cuda()
    with torch.no_grad():
        for j in range(samples):
            outputs[j] = net(im)
    score = outputs.mean(0)
    probs = F.softmax(score, dim=1).cpu()
    # entropy map
    confidence_map = (probs * torch.log(probs + 1e-9)).sum(dim=1)  # entropy
    # 1 vs all entropy
    confidence = torch.zeros_like(probs) #.cuda()
    for j in range(opt.label_nc):
        confidence[:,j,:,:] = probs[:,j,:,:] * torch.log(probs[:,j,:,:]+1e-9) + \
                        probs[:,torch.arange(opt.label_nc)!=j,:,:].sum(dim=1) * \
            torch.log(probs[:,torch.arange(opt.label_nc)!=j,:,:].sum(dim=1)+1e-9)
    confidence = confidence.mean(dim=(2,3))
    _, preds = torch.max(score, 1)


    hist = fast_hist(label.numpy().flatten(),
            preds.cpu().numpy().flatten(),
            19)
    acc_overall, acc_percls, iu, fwIU, pix_percls = result_stats(hist)
    iterations.set_postfix({'mIoU':' {:0.2f}  fwIoU: {:0.2f} pixel acc: {:0.2f} per cls acc: {:0.2f}'.format(
        np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))})
    metric = [iu.tolist(), pix_percls.tolist(), fwIU, acc_overall, acc_percls.tolist(), confidence[0].numpy().tolist()]
    metrics.append(metric)
    if opt.phase == 'train':
        output_dir = osp.join(opt.eval_output_dir, 'metrics_trainccv_mcd')
    else:
        output_dir = osp.join(opt.eval_output_dir, 'metrics_mcd_val')
        conf_path = osp.join(opt.eval_output_dir, data_i['label_path'][0].replace('gtFine', 'gtFinePred_mcdropout'))
        os.makedirs(os.path.dirname(conf_path), exist_ok=True)
        np.savez_compressed(conf_path, confidence_map=confidence_map.cpu().numpy()[0])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, os.path.splitext(os.path.basename(data_i['path'][0]))[0] + '.json'), 'w') as f:
        json.dump(metric, f)
print()
