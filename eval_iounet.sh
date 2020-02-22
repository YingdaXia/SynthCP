GPUs=$1
eval_output_dir=$2
IOUNET_NAME=$3
for iter in {5000..5000..5000}
do
echo $iter
python eval_iounet_v2.py --name cityscapes --dataset_mode iou \
                --phase test --n_fold 0 \
                --dataroot ./cityscapes \
                --image_src_dir ./$eval_output_dir/cityscapes/leftImg8bitResize/val \
                --image_rec_dir ./$eval_output_dir/cityscapes/leftImg8bitRec/val \
                --iou_dir ./$eval_output_dir/metrics_val \
                --pred_dir ./$eval_output_dir/cityscapes/gtFinePredProb/val \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --model_path checkpoints/iounet/$IOUNET_NAME/iter$iter.pth \
                --eval_iter $iter \
                --use_vae \
                --vgg_norm \
                --gpu_ids $GPUs
done
