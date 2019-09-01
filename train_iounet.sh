GPUs=$1
python train_iounet.py --name cityscapes_iounet_c19 --dataset_mode iou \
                --dataroot /data/yzhang/cityscapes \
                --image_src_dir /data/yzhang/cityscapes/leftImg8bit/val \
                --image_rec_dir /data/yzhang/cityscapes/leftImg8bit/val \
                --iou_dir /data/yzhang/Domain-Adaptation/metrics \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 4 \
                --lr 2e-4 \
                --gpu_ids $GPUs

                #--image_src_dir /data/yzhang/cityscapes/SrcImgResize/train \
                #--image_rec_dir /data/yzhang/cityscapes/RecImgResize/train \
