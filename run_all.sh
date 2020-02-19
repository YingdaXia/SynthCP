EXP_NAME=deeplab_lr_newspade
#IOUNET_NAME=cityscapes_iouconf_hce2_1_concatInput_concatNet_reproduce2
IOUNET_NAME=cityscapes_iouconf_hce2_1_concatInput_concatNet_newspade_dl
#IOUNET_NAME=cityscapes_iouconf_hce2_1_baseline_reproduce2
#bash train_fcn.sh 1
#bash train_fcn_ccv.sh 0 0 &
#bash train_fcn_ccv.sh 1 1 &
#bash train_fcn_ccv.sh 2 2 &
#bash train_fcn_ccv.sh 3 3 &
#wait
#bash eval_fcn_single.sh 0 0 $EXP_NAME &
#bash eval_fcn_single.sh 1 1 $EXP_NAME &
#bash eval_fcn_single.sh 2 2 $EXP_NAME &
#bash eval_fcn_single.sh 3 3 $EXP_NAME &
#wait
#bash eval_fcn_single_test.sh 0 $EXP_NAME

#bash eval_deeplab_single.sh 0 0 $EXP_NAME &
#bash eval_deeplab_single.sh 1 1 $EXP_NAME &
#bash eval_deeplab_single.sh 2 2 $EXP_NAME &
#bash eval_deeplab_single.sh 3 3 $EXP_NAME &
#wait
#bash eval_deeplab_single_test.sh 2 $EXP_NAME
#
bash eval_spade.sh 2 $EXP_NAME 200 train &
bash eval_spade.sh 3 $EXP_NAME 200 test &
wait
bash train_iounet.sh 3 $EXP_NAME $IOUNET_NAME
bash eval_iounet.sh 3 $EXP_NAME $IOUNET_NAME


