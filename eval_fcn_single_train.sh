GPUs=$1
eval_output_dir=$2

python eval_fcn_single.py --name cityscapes --dataset_mode cityscapes \
                --phase train --cross_validation_mode train --n_fold 0 --fold 0 \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot ./cityscapes \
                --model_path checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
                --use_vae \
                --vgg_norm \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \
                #--load_size 1024 \
                #--crop_size 1024
