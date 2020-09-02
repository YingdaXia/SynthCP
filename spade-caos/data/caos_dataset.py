"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import numpy as np
import torch
import os

class CaosDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        #parser.set_defaults(preprocess_mode='scale_width')
        #parser.set_defaults(load_size=640)
        #parser.set_defaults(crop_size=640)
        #parser.set_defaults(display_winsize=640)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=13)
        #parser.set_defaults(aspect_ratio=1.777)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt, adda_mode = 'normal'):
        root = opt.dataroot
        if opt.eval_spade:
            label_dir = opt.label_dir
        else:
            if opt.phase == 'train':
                label_dir = os.path.join(root, 'train', 'annotations', 'training')
            elif opt.phase == 'val':
                label_dir = os.path.join(root, 'train', 'annotations', 'validation')
            elif opt.phase == 'test':
                label_dir = os.path.join(root, 'test', 'annotations', 'test')
        label_paths_all = make_dataset(label_dir, recursive=True)
        label_paths = [p for p in label_paths_all if p.endswith('.png')]

        if opt.phase == 'train':
            image_dir = os.path.join(root, 'train', 'images', 'training')
        elif opt.phase == 'val':
            image_dir = os.path.join(root, 'train', 'images', 'validation')
        elif opt.phase == 'test':
            image_dir = os.path.join(root, 'test', 'images', 'test')
        image_paths = make_dataset(image_dir, recursive=True)

        instance_paths = []

        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        folder1, name1 = path1.split('/')[-2:]
        folder2, name2 = path2.split('/')[-2:]

        return folder1 == folder2 and name1 == name2

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)

        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0 - 1.0

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # for VGG segmentation network
        transform_image_vgg = get_transform(self.opt, params, for_VGG=True)
        image_tensor_vgg = transform_image_vgg(image)

        # label remapping
        """
        if not self.opt.eval_spade:
            label_tensor_transform = self.ignore_label * torch.ones_like(label_tensor)
            for k,v in self.id_to_trainid.items():
                label_tensor_transform[label_tensor == k] = v
            label_tensor = label_tensor_transform
            label_tensor[label_tensor == self.ignore_label] = self.opt.label_nc  # 'unknown' is opt.label_nc
            """

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                    'instance': instance_tensor,
                    'image': image_tensor,
                    'image_seg':image_tensor_vgg,
                    'path': image_path,
                    'label_path': label_path,
                    }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict