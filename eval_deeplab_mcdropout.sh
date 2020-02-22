fold=$1
GPUs=$2
eval_output_dir=$3
python eval_deeplab_mcdropout.py --name cityscapes --dataset_mode cityscapes \
                --phase train --cross_validation_mode val --n_fold 4 --fold $1 \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot ./cityscapes \
                --model_path checkpoints/deeplab/cityscapes_c19_$fold-iter50000.pth \
                --use_vae \
                --vgg_norm \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \
                #--load_size 1024 \
                #--crop_size 1024
