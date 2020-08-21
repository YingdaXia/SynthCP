GPUs=$1
eval_output_dir=$2
IOUNET_NAME=$3
REC_PATH=$4
#cityscapes_iouconf_hce2_1_concatInput_concatNet
python tools/train_iounet_v2.py --name $IOUNET_NAME --dataset_mode iou \
                --dataroot ./cityscapes \
                --image_src_dir ./$eval_output_dir/cityscapes/leftImg8bitResize/train \
                --image_rec_dir ./$eval_output_dir/cityscapes/$REC_PATH/train \
                --iou_dir ./$eval_output_dir/metrics_trainccv \
                --pred_dir ./$eval_output_dir/cityscapes/gtFinePredProb/train \
                --label_nc 19 --no_instance \
                --vgg_norm \
                --batchSize 4 \
                --lr 2e-4 \
                --niter 40000 \
                --snapshot 2000 \
                --gpu_ids $GPUs --nThreads 8
