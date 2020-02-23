GPUs=$1
python train_deeplab.py --name cityscapes_c19 --dataset_mode cityscapes \
                --dataroot ./cityscapes \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 8 \
                --lr 2e-4 \
                --n_fold 0 \
                --gpu_ids $GPUs \
                --niter 50000
                #--load_size 1024 \
                #--crop_size 1024
