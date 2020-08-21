import os.path
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collections import deque

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import json, pickle
from tqdm import tqdm
from scipy import stats

from options.iounet_options import BaseOptions
from util.eval_util import Metrics, eval_ood_measure, eval_alarm_metrics
import data
import pdb

def main():
    # parse options
    opt = BaseOptions().parse()

    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader(opt)

    aurocs, auprs, fprs = [], [], []
    pred_ious, real_ious = [], []

    metrics = Metrics(
            #['accuracy', 'mean_iou', 'auc', 'ap_success', 'ap_errors', 'fpr_at_95tpr'], 500 * 256 * 512, 19
            ['accuracy', 'mean_iou', 'auc', 'ap_success', 'ap_errors'], 500 * 256 * 512, 19
            )
    for i, data_i in tqdm(enumerate(dataloader)):

        im_src = data_i['image_src'].cuda()
        im_rec = data_i['image_rec'].cuda()
        conf = data_i['conf_map'].cuda()

        iou_label = data_i['iou'].cuda()
        prob = data_i['prob'].cuda()
        label_map = data_i['label_map'].cuda()
        max_prob, pred = torch.nn.Softmax(dim=1)(prob).max(dim=1)

        pred = pred[label_map != 19]
        conf = conf[label_map != 19]
        max_prob = max_prob[label_map != 19]
        label_map = label_map[label_map != 19]

        res = eval_ood_measure(conf.cpu().numpy(), pred.cpu().numpy(), label_map.cpu().numpy(), mask=None)
        if res is not None:
            auroc, aupr, fpr = res
            aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)

        metrics.update(pred.long(), label_map.long(), conf)

    print(" mean fpr = ", np.mean(fprs))

    scores = metrics.get_scores(split="val")
    logs_dict = {}
    for s in scores:
        logs_dict[s] = scores[s]
    print(logs_dict)


if __name__ == '__main__':
    main()
