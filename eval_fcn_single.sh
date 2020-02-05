fold=$1
GPUs=$2
python eval_fcn_single.py --name cityscapes --dataset_mode cityscapes \
                --phase train --cross_validation_mode val --n_fold 4 --fold $1 \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot /mnt/sdd/yingda/data/alarmseg/cityscapes \
                --model_path checkpoints/fcn8s/cityscapes_c19_$fold-iter100000.pth \
                --use_vae \
                --vgg_norm \
                --gpu_ids $GPUs
#--model_path /mnt/sdd/yingda/alarmseg-spade/checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
#--model_path /data/yingda/alarmseg-spade/checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
#--model_path checkpoints/fcn8s/cityscapes_c19_$fold-iter100000.pth \
