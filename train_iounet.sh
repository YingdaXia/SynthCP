GPUs=$1
python train_iounet_v2.py --name cityscapes_iouconf_hce2_1 --dataset_mode iou \
                --dataroot mnt/sdd/yingda/data/alarmseg/cityscapes \
                --image_src_dir /mnt/sdd/yingda/data/alarmseg/cityscapes/leftImg8bitResize/train \
                --image_rec_dir /mnt/sdd/yingda/data/alarmseg/cityscapes/leftImg8bitRec/train \
                --iou_dir /mnt/sdd/yingda/alarmseg-spade/metrics_trainccv \
                --pred_dir /mnt/sdd/yingda/data/alarmseg/cityscapes/gtFinePredProb/train \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 4 \
                --lr 2e-4 \
                --niter 20000 \
                --snapshot 5000 \
                --gpu_ids $GPUs

                #--image_src_dir /data/yzhang/cityscapes/SrcImgResize/train \
                #--image_rec_dir /data/yzhang/cityscapes/RecImgResize/train \
