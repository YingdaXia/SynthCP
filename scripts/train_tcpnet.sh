MODEL_NAME=$1
GPUs=$2
eval_output_dir=$3
IOUNET_NAME=$4
#cityscapes_iouconf_hce2_1_concatInput_concatNet
python tools/train_tcpnet.py --name $IOUNET_NAME --dataset_mode iou \
                --dataroot ./$eval_output_dir/cityscapes \
                --image_src_dir ./$eval_output_dir/cityscapes/leftImg8bitResize/train \
                --image_rec_dir ./$eval_output_dir/cityscapes/leftImg8bitRec/train \
                --iou_dir ./$eval_output_dir/metrics_trainccv \
                --pred_dir ./$eval_output_dir/cityscapes/gtFinePredProb/train \
                --model_path checkpoints/fcn8s/${MODEL_NAME}-iter100000.pth \
                --label_nc 19 --no_instance \
                --vgg_norm \
                --batchSize 4 \
                --lr 5e-5 \
                --niter 100000 \
                --snapshot 5000 \
                --gpu_ids $GPUs --nThreads 8
