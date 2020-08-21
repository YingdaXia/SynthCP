EXP_NAME=deeplab_exp
SEG_MODEL_NAME=cityscapes_c19
EXP_PATH=exps/$EXP_NAME
LINREG_NAME=mcdropout_linreg_deeplab
LINREG_MODEL_PATH=checkpoints/iounet/$LINREG_NAME/iter20000.pth

mkdir -p $EXP_PATH
mkdir -p $EXP_PATH/cityscapes
ln -s `pwd`/datasets/cityscapes/leftImg8bit $EXP_PATH/cityscapes/
ln -s `pwd`/datasets/cityscapes/gtFine $EXP_PATH/cityscapes/

# Run MCDropout
# bash scripts/eval_deeplab_mcdropout.sh $SEG_MODEL_NAME 0 0 $EXP_PATH &
# bash scripts/eval_deeplab_mcdropout.sh $SEG_MODEL_NAME 1 1 $EXP_PATH &
# bash scripts/eval_deeplab_mcdropout.sh $SEG_MODEL_NAME 2 2 $EXP_PATH &
# bash scripts/eval_deeplab_mcdropout.sh $SEG_MODEL_NAME 3 3 $EXP_PATH &
# wait
bash scripts/eval_deeplab_mcdropout_test.sh $SEG_MODEL_NAME 0 $EXP_PATH

# Image-level failure detection
#### If you want to retrain the linear regression
# bash scripts/train_linreg.sh 0 $EXP_PATH $LINREG_NAME
#### test pretrained model
bash scripts/eval_linreg.sh $LINREG_MODEL_PATH 0 $EXP_PATH

# Pixel-level failure detection
bash scripts/eval_mcdropout.sh 0 $EXP_PATH

