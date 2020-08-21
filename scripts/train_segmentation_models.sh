EXP_NAME=fcn_new_exp
MODEL_NAME=cityscapes_c19_reproduce

# prepare exp path
EXP_PATH=exps/$EXP_NAME
mkdir -p $EXP_PATH
mkdir -p $EXP_PATH/cityscapes
ln -s `pwd`/datasets/cityscapes/leftImg8bit $EXP_PATH/cityscapes/
ln -s `pwd`/datasets/cityscapes/gtFine $EXP_PATH/cityscapes/

# train FCN8s using cross cross validation
bash scripts/train_fcn_ccv.sh $MODEL_NAME 0 0 &
bash scripts/train_fcn_ccv.sh $MODEL_NAME 1 1 &
bash scripts/train_fcn_ccv.sh $MODEL_NAME 2 2 &
bash scripts/train_fcn_ccv.sh $MODEL_NAME 3 3 &
wait
bash scripts/train_fcn.sh $MODEL_NAME 0

# train Deeplabv2 using cross cross validation
# bash scripts/train_deeplab_ccv.sh $MODEL_NAME 0 0 &
# bash scripts/train_deeplab_ccv.sh $MODEL_NAME 1 1 &
# bash scripts/train_deeplab_ccv.sh $MODEL_NAME 2 2 &
# bash scripts/train_deeplab_ccv.sh $MODEL_NAME 3 3 &
# wait
# bash scripts/train_deeplab.sh $MODEL_NAME 0
