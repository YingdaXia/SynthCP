GPUs=$1
eval_output_dir=$2
python tools/eval_deeplab_single.py --name cityscapes --dataset_mode cityscapes \
                --phase test --cross_validation_mode train --n_fold 0 --fold 0 \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --dataroot ./$eval_output_dir/cityscapes \
                --model_path checkpoints/deeplab/cityscapes_c19-iter50000.pth \
                --vgg_norm \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \
