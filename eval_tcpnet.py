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
from options.iounet_options import BaseOptions

from models.fcn8_self_confid import VGG16_FCN8s_SelfConfid
import anom_utils
from metric import Metrics
import data
import json, pickle
from tqdm import tqdm
from scipy import stats
import pdb

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

def eval_ood_measure(conf, pred, seg_label, mask=None):
    correct_map = pred == seg_label
    out_label = np.logical_not(correct_map)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return None

def eval_alarm_metrics(pred_ious, real_ious):
    mae = np.nanmean(np.abs(np.array(pred_ious) - np.array(real_ious)), axis=0)
    std = np.nanstd(np.abs(np.array(pred_ious) - np.array(real_ious)), axis=0)
    classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                            'bicycle']
    pred_ious = np.array(pred_ious)
    real_ious = np.array(real_ious)
    pcs = []
    scs = []
    for cls, cls_name in enumerate(classes):
        valid_inds = np.logical_not(np.isnan(real_ious[:,cls]))
        pc, p = stats.pearsonr(real_ious[:,cls][valid_inds], pred_ious[:,cls][valid_inds])
        sc, p = stats.spearmanr(real_ious[:,cls][valid_inds], pred_ious[:,cls][valid_inds])
        #print('%s, correlation coefficient: %.3f, p value: %.6f' % (cls_name, r, p))
        pcs.append(pc)
        scs.append(sc)
    print(("{},"*len(classes)).format(*classes))
    print(("P.C. = "+"{:.6f},"*len(classes)).format(*pcs))
    print("mean P.C. = ", np.nanmean(pcs))
    print(("S.C. = "+"{:.6f},"*len(classes)).format(*scs))
    print("mean S.C. = ", np.nanmean(scs))
    print(("MAE = "+"{:.6f},"*len(classes)).format(*mae))
    print("mmae = ", np.nanmean(mae))
    print(("STD = "+"{:.6f},"*len(classes)).format(*std))
    print("mstd = ", np.nanmean(std))

    print('copy paste')
    print(("{:.6f},"*len(classes)).format(*pcs))
    print(("{:.6f},"*len(classes)).format(*scs))
    print(("{:.6f},"*len(classes)).format(*mae))
    print(("{:.6f},"*len(classes)).format(*std))

def main():
    # parse options
    opt = BaseOptions().parse()

    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader(opt)

    #net = IOUwConfNetBaseline(num_cls=opt.label_nc)
    #net = IOUwConfNet(num_cls=opt.label_nc)

    net = VGG16_FCN8s_SelfConfid(num_cls=opt.label_nc, pretrained=False)
    net.load_state_dict(torch.load(opt.model_path, map_location='cuda:{}'.format(opt.gpu_ids[0])))
    net.eval()
    net.cuda()
    transform = []
    target_transform = []

    iteration = 0
    losses = deque(maxlen=10)
    aurocs, auprs, fprs = [], [], []
    pred_ious, real_ious = [], []

    metrics = Metrics(
            #['accuracy', 'mean_iou', 'auc', 'ap_success', 'ap_errors', 'fpr_at_95tpr'], 500 * 256 * 512, 19
            ['accuracy', 'mean_iou', 'auc', 'ap_success', 'ap_errors'], 500 * 256 * 512, 19
            )
    loss, len_steps, len_data = 0, 0, 0
    for i, data_i in tqdm(enumerate(dataloader)):
        # Clear out gradients

        # load data/label
        #im = make_variable(im, requires_grad=False)
        #label = make_variable(label, requires_grad=False)

        # forward pass and compute loss
        im_src = data_i['image_src'].cuda()

        iou_label = data_i['iou'].cuda()
        prob = data_i['prob'].cuda()
        label_map = data_i['label_map'].cuda()
        #pred = prob.argmax(dim=1)
        max_prob, pred = torch.nn.Softmax(dim=1)(prob).max(dim=1)

        with torch.no_grad():
            _, conf = net(im_src)
        #pdb.set_trace()

        len_steps += 1 * 256 * 512
        len_data += 1

        pred = pred[label_map != 19]
        conf = conf.squeeze(0)
        conf = conf[label_map != 19]
        max_prob = max_prob[label_map != 19]
        label_map = label_map[label_map != 19]


        #correct_map = (pred.long() == label_map.long()).float()
        #tensor_max = torch.abs((conf-max_prob) / 2) + torch.abs((conf + max_prob) / 2)
        #tensor_min = -1.0 * torch.abs((conf-max_prob) / 2) + torch.abs((conf + max_prob) / 2)
        #conf = tensor_max * correct_map + tensor_min * (1 - correct_map)
        #conf = conf + max_prob
        #conf = max_prob
        metrics.update(pred.long(), label_map.long(), conf)
        #metrics.update(pred.long(), label_map.long(), max_prob)

        #valid=data_i['valid'][0].cpu().numpy()
        #real_iou_valid = iou_label[0].cpu().numpy() / 100
        #real_iou_valid[valid==0] = np.nan
        #pred_ious.append(pred_iou[0].cpu().numpy())
        #real_ious.append(real_iou_valid)

        #conf_dir = os.path.join('./checkpoints', opt.name, 'confnetpred')
        #os.makedirs(conf_dir, exist_ok=True)
        #np.savez_compressed(os.path.join(conf_dir, os.path.basename(data_i['image_src_path'][0])),
        #                    conf=conf[0].cpu().numpy(), prob=prob[0].cpu().numpy(), label=label_map[0].cpu().numpy())

        #metric = [pred_iou.cpu().numpy()[0].tolist()]
        #opt.metric_pred_dir = os.path.join('./checkpoints', opt.name, 'metrics_pred_iouconf')

        #os.makedirs(opt.metric_pred_dir, exist_ok=True)
        #with open(os.path.join(opt.metric_pred_dir, os.path.splitext(os.path.basename(data_i['image_src_path'][0]))[0] + '.json'), 'w') as f:
        #    json.dump(metric, f)
    scores = metrics.get_scores(split="val")
    logs_dict = {}
    for s in scores:
        logs_dict[s] = scores[s]
    print(logs_dict)

    #with open(osp.join(osp.dirname(opt.model_path), 'iou_pred_iter{}.pkl'.format(opt.eval_iter)), 'wb') as f:
    #    pickle.dump({'pred_ious':pred_ious, 'real_ious':real_ious}, f)

    #eval_alarm_metrics(pred_ious, real_ious)

if __name__ == '__main__':
    main()
