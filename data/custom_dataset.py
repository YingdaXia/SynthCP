"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=19)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt, adda_mode = 'normal'):
        if adda_mode == 'normal':
            label_dir = opt.label_dir
            image_dir = opt.image_dir
        elif adda_mode == 'source':
            label_dir = opt.label_dir_source
            image_dir = opt.image_dir_source
        elif adda_mode == 'target':
            label_dir = opt.label_dir_target
            image_dir = opt.image_dir_target
        
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        instance_paths = []
        
        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths


class AddaDataset(torch.utils.data.Dataset):

    def __init__(self, src_data, tgt_data):
        self.src = src_data
        self.tgt = tgt_data

    def __getitem__(self, index):
        ns = len(self.src)
        nt = len(self.tgt)
        data_s = self.src[index % ns]
        data_t = self.tgt[index % nt]
        return data_s, data_t

    def __len__(self):
        return max(len(self.src), len(self.tgt))
