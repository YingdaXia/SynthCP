GPUs=$1
eval_output_dir=$2
MODEL_PATH=$3
REC_PATH=$4

python tools/eval_iounet_v2.py --name cityscapes --dataset_mode iou \
                --phase test --n_fold 0 \
                --dataroot ./cityscapes \
                --image_src_dir ./$eval_output_dir/cityscapes/leftImg8bitResize/val \
                --image_rec_dir ./$eval_output_dir/cityscapes/$REC_PATH/val \
                --iou_dir ./$eval_output_dir/metrics_val \
                --pred_dir ./$eval_output_dir/cityscapes/gtFinePredProb/val \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --model_path $MODEL_PATH \
                --vgg_norm \
                --gpu_ids $GPUs
