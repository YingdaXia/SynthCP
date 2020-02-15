GPUs=$1
python train_linreg.py --name cityscapes_iouconf_mcdropout --dataset_mode iouentropy \
                --dataroot ./cityscapes \
                --image_src_dir ./cityscapes/leftImg8bitResize_hr/train \
                --image_rec_dir ./cityscapes/leftImg8bitRecVAE256E300/train \
                --iou_dir ./metrics_hr_trainccv \
                --entropy_dir ./metrics_hr_trainccv_mcd \
                --pred_dir ./cityscapes/gtFinePredProb_hr/train \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 4 \
                --lr 2e-4 \
                --niter 20000 \
                --snapshot 5000 \
                --gpu_ids $GPUs

