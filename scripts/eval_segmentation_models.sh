EXP_NAME=fcn_exp
MODEL_NAME=cityscapes_c19

# prepare exp path
EXP_PATH=exps/$EXP_NAME
mkdir -p $EXP_PATH
mkdir -p $EXP_PATH/cityscapes
ln -s `pwd`/datasets/cityscapes/leftImg8bit $EXP_PATH/cityscapes/
ln -s `pwd`/datasets/cityscapes/gtFine $EXP_PATH/cityscapes/

# evaluate FCN8s 
# bash scripts/eval_fcn_single.sh $MODEL_NAME 0 0 $EXP_PATH &
# bash scripts/eval_fcn_single.sh $MODEL_NAME 1 1 $EXP_PATH &
# bash scripts/eval_fcn_single.sh $MODEL_NAME 2 2 $EXP_PATH &
# bash scripts/eval_fcn_single.sh $MODEL_NAME 3 3 $EXP_PATH &
# wait
bash scripts/eval_fcn_single_test.sh $MODEL_NAME 0 $EXP_PATH

# evaluate Deeplabv2
# bash scripts/eval_deeplab_single.sh $MODEL_NAME 0 0 $EXP_PATH &
# bash scripts/eval_deeplab_single.sh $MODEL_NAME 1 1 $EXP_PATH &
# bash scripts/eval_deeplab_single.sh $MODEL_NAME 2 2 $EXP_PATH &
# bash scripts/eval_deeplab_single.sh $MODEL_NAME 3 3 $EXP_PATH &
# wait
# bash scripts/eval_deeplab_single_test.sh $MODEL_NAME 0 $EXP_PATH 
