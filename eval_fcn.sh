python eval_fcn.py --name cityscapes_from_GTA5 --dataset_mode cityscapes \
                --phase test \
                --label_nc 19 --no_instance \
                --dataroot /data/yzhang/cityscapes \
                --use_vae \
                --vgg_norm \
                --model_path /data/yingda/Domain-Adaptation/checkpoints/fcn8s/cityscapes_from_GTA5_c19-iter50000.pth \
                --gpu_ids 3
#--model_path checkpoints/fcn8s/cityscapes_from_GTA5-iter100000.pth \
