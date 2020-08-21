MODEL_NAME=$1
GPUs=$2
eval_output_dir=$3
python tools/eval_fcn_mcdropout.py --name cityscapes --dataset_mode cityscapes \
                --phase test --cross_validation_mode train --n_fold 0 --fold 0 \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot ./cityscapes \
                --model_path checkpoints/fcn8s/${MODEL_NAME}-iter100000.pth \
                --vgg_norm \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \
                #--load_size 1024 \
                #--crop_size 1024

