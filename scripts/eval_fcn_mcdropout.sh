MODEL_NAME=$1
fold=$2
GPUs=$3
eval_output_dir=$4
python tools/eval_fcn_mcdropout.py --name cityscapes --dataset_mode cityscapes \
                --phase train --cross_validation_mode val --n_fold 4 --fold $fold \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot ./cityscapes \
                --model_path checkpoints/fcn8s/${MODEL_NAME}_$fold-iter100000.pth \
                --vgg_norm \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \