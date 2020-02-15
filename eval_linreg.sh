GPUs=$1
for iter in {5000..20000..5000}
do
#iter=15000
echo $iter
python eval_linreg.py --name cityscapes --dataset_mode iouentropy \
                --phase test --n_fold 0 \
                --dataroot ./cityscapes \
                --image_src_dir ./cityscapes/leftImg8bitResize/val \
                --image_rec_dir ./cityscapes/leftImg8bitRecVAE256E300/val \
                --entropy_dir ./metrics_trainccv_mcd \
                --iou_dir ./metrics_val \
                --pred_dir ./cityscapes/gtFinePredProb_hr/val \
                --label_nc 19 --no_instance --serial_batches --no_flip \
                --model_path checkpoints/iounet/cityscapes_iouconf_mcdropout/iter$iter.pth \
                --eval_iter $iter \
                --use_vae \
                --vgg_norm \
                --gpu_ids $GPUs
done
