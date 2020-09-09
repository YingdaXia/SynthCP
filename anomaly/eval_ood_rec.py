# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset import ValDataset
from models import ModelBuilder
from models import SegmentationModuleOOD as SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

import anom_utils

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))

def eval_ood_measure(conf, seg_label, cfg, mask=None):
    out_labels = cfg.OOD.out_labels
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return None


def evaluate(segmentation_module, loader, loader_rec, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    aurocs, auprs, fprs = [], [], []

    pbar = tqdm(total=len(loader))
    for batch_data, rec_data in zip(loader, loader_rec):
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        #print(batch_data['name'])
        full_name = batch_data['name']
        img_folder, img_name = batch_data['name'].split('/')
        del batch_data['name']

        rec_data = rec_data[0]
        seg_label_rec = as_numpy(rec_data['seg_label'][0])
        img_resized_list_rec = rec_data['img_data']
        #print(rec_data['name'])
        del rec_data['name']


        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            ft1 = torch.zeros(1, 4096, int(segSize[0] / 4), int(segSize[1] / 4))
            ft2 = torch.zeros(1, 4096, int(segSize[0] / 4), int(segSize[1] / 4))

            scores = async_copy_to(scores, gpu)
            ft1 = async_copy_to(ft1, gpu)
            ft2 = async_copy_to(ft2, gpu)
            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp, ft_temp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                ft_temp = nn.functional.interpolate(ft_temp, size=ft1.shape[2:], mode='bilinear', align_corners=False)
                ft1 = ft1 + ft_temp / len(cfg.DATASET.imgSizes)

            for img in img_resized_list_rec:
                feed_dict = rec_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                _, ft_temp = segmentation_module(feed_dict, segSize=segSize)
                ft_temp = nn.functional.interpolate(ft_temp, size=ft2.shape[2:], mode='bilinear', align_corners=False)
                ft2 = ft2 + ft_temp / len(cfg.DATASET.imgSizes)

            tmp_scores = scores
            if cfg.OOD.exclude_back:
                tmp_scores = tmp_scores[:,1:]


            mask = None
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

            #for evaluating MSP
            if cfg.OOD.ood == "msp":
                conf, _  = torch.max(tmp_scores, dim=1)
                conf = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "rec":
                msp, _  = torch.max(tmp_scores, dim=1)
                msp = msp.squeeze(0).cpu()
                ft1 = nn.functional.normalize(ft1, dim=1)
                ft2 = nn.functional.normalize(ft2, dim=1)
                #ft_dist = torch.nn.SmoothL1Loss(reduction='none')(ft1, ft2)
                ft_dist = nn.functional.cosine_similarity(ft1, ft2, dim=1).unsqueeze(1)
                ft_dist = nn.functional.interpolate(ft_dist, size=segSize, mode='bilinear', align_corners=False)[0,0].cpu()
                conf_rec = ft_dist
                t = 0.999
                conf = msp * (msp > t).float() + conf_rec * (msp <= t).float()
                #conf = 1- (1 - conf_rec) * (msp <= 0.99).float()

                conf = as_numpy(conf.squeeze(0).cpu())

            res = eval_ood_measure(conf, seg_label, cfg, mask=mask)
            if res is not None:
                auroc, aupr, fpr = res
                aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)
            else:
                pass

            res1 = eval_ood_measure(msp, seg_label, cfg, mask=mask)
            if res is not None:
                auroc1, aupr1, fpr1 = res1
                #aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)
            else:
                pass


        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)
        torch.cuda.empty_cache()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))
    print("mean auroc = ", np.mean(aurocs), "mean aupr = ", np.mean(auprs), " mean fpr = ", np.mean(fprs))

def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    data_set_rec = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        rec_dataset=cfg.DATASET.rec_dataset)

    loader_rec = torch.utils.data.DataLoader(
        data_set_rec,
        batch_size=cfg.VAL.batch_size,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, loader_rec, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/test_ood_rec.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "--ood",
        help="Choices are [msp, crf-gauss, crf, maxlogit, background]",
        default="rec",
    )
    parser.add_argument(
        "--exclude_back",
        help="Whether to exclude the background class.",
        action="store_true",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    ood = ["OOD.exclude_back", args.exclude_back, "OOD.ood", args.ood]

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(ood)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)
