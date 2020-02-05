GPUs=$1
python eval_iounet_v2.py --name cityscapes --dataset_mode iou \
                --phase test --n_fold 0 \
                --dataroot mnt/sdd/yingda/data/alarmseg/cityscapes \
                --image_src_dir /mnt/sdd/yingda/data/alarmseg/cityscapes/leftImg8bitResize/val \
                --image_rec_dir /mnt/sdd/yingda/data/alarmseg/cityscapes/leftImg8bitRec/val \
                --iou_dir /mnt/sdd/yingda/alarmseg-spade/metrics_val \
                --pred_dir /mnt/sdd/yingda/data/alarmseg/cityscapes/gtFinePredProb/val \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --model_path checkpoints/iounet/cityscapes_iounet_c19-iter100000.pth \
                --use_vae \
                --vgg_norm \
                --gpu_ids $GPUs
#--model_path checkpoints/fcn8s/cityscapes_from_GTA5-iter100000.pth \
#--model_path /data/yingda/Domain-Adaptation/checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
#--model_path checkpoints/fcn8s/cityscapes_c19_$fold-iter100000.pth \
