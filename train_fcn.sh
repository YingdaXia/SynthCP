python train_fcn.py --name cityscapes_from_GTA5 --dataset_mode custom \
                --label_dir /data/yzhang/gta5_deeplab/labels \
                --image_dir /data/yzhang/gta5_deeplab/images \
                --label_nc 35 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 8 \
                --lr 2e-4 \
                --gpu_ids 7