#python train_fcn.py --name cityscapes_from_GTA5_c19 --dataset_mode custom \
#                --label_dir /data/yzhang/gta5_deeplab/labels \
#                --image_dir /data/yzhang/gta5_deeplab/images \
#                --label_nc 19 --no_instance \
#                --use_vae \
#                --vgg_norm \
#                --batchSize 8 \
#                --lr 2e-4 \
#                --gpu_ids 1
python train_fcn.py --name cityscapes_c19 --dataset_mode cityscapes \
                --dataroot /data/yzhang/cityscapes \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 8 \
                --lr 2e-4 \
                --gpu_ids 2