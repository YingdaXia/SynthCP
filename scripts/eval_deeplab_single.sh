MODEL_NAME=$1
fold=$2
GPUs=$3
eval_output_dir=$4
python tools/eval_deeplab_single.py --name cityscapes --dataset_mode cityscapes \
                --phase train --cross_validation_mode val --n_fold 4 --fold $fold \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot ./$eval_output_dir/cityscapes \
                --model_path checkpoints/deeplab/${MODEL_NAME}_$fold-iter50000.pth \
                --vgg_norm \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \
