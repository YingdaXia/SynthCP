#python train_fcn.py --name cityscapes_from_GTA5_c19 --dataset_mode custom \
#                --label_dir /data/yzhang/gta5_deeplab/labels \
#                --image_dir /data/yzhang/gta5_deeplab/images \
#                --label_nc 19 --no_instance \
#                --use_vae \
#                --vgg_norm \
#                --batchSize 8 \
#                --lr 2e-4 \
#                --gpu_ids 1

fold=$1
GPUs=$2
#cityscapes_c19_$fold
python train_fcn.py --name cityscapes_hr_c19_$fold --dataset_mode cityscapes \
                --dataroot ./cityscapes \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 4 \
                --lr 2e-4 \
                --n_fold 4 --fold $fold \
                --gpu_ids $GPUs \
                --snapshot 10000 \
                --load_size 1024 \
                --crop_size 1024

