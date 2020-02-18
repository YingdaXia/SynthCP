GPUs=$1
python train_iounet_v2.py --name cityscapes_iouconf_hce2_1_concatInput_concatNet --dataset_mode iou \
                --dataroot ./cityscapes \
                --image_src_dir ./cityscapes/leftImg8bitResize/train \
                --image_rec_dir ./cityscapes/leftImg8bitRecNoVAE50/train \
                --iou_dir ./metrics_trainccv \
                --pred_dir ./cityscapes/gtFinePredProb/train \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 4 \
                --lr 2e-4 \
                --niter 20000 \
                --snapshot 5000 \
                --gpu_ids $GPUs --nThreads 8

                #--image_rec_dir ./cityscapes/leftImg8bitRecVAE256E300/train \
