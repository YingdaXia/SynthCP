GPUs=$1
python eval_iounet.py --name cityscapes --dataset_mode iou \
                --phase test --n_fold 0 \
                --image_src_dir /data/yzhang/cityscapes/leftImg8bitResize/val \
                --image_rec_dir /data/yzhang/cityscapes/leftImg8bitRec/val \
                --iou_dir /data/yzhang/Domain-Adaptation/metrics_val \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot /data/yzhang/cityscapes \
                --model_path checkpoints/fcn8s/cityscapes_iounet_c19-iter50000.pth \
                --use_vae \
                --vgg_norm \
                --gpu_ids $GPUs
#--model_path checkpoints/fcn8s/cityscapes_from_GTA5-iter100000.pth \
#--model_path /data/yingda/Domain-Adaptation/checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
#--model_path checkpoints/fcn8s/cityscapes_c19_$fold-iter100000.pth \
