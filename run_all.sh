#REC_PATH=leftImg8bitRecNew200
REC_PATH=leftImg8bitRec
#REC_PATH=leftImg8bitRecVAE256E200
#REC_PATH=rec_pix2pixhd
#EXP_NAME=fcn_lr
#IOUNET_NAME=cityscapes_iouconf_hce2_1_concatInput_concatNet_10iouloss #_more #_newspade
#IOUNET_NAME=cityscapes_iouconf_hce2_1_concatInput_concatNet #_more #_newspade
#IOUNET_NAME=cityscapes_iouconf_hce2_1_baseline_reproduce2 #_more #_newspade
#IOUNET_NAME=cityscapes_iou_hce2_1_concatInput_concatNet #_more #_newspade
#IOUNET_NAME=cityscapes_conf_hce2_1_concatInput_concatNet #_more #_newspade
#IOUNET_NAME=cityscapes_iouconf_hce2_1_concatInput_concatNet_p2phd #_more #_newspade

EXP_NAME=deeplab_lr
#IOUNET_NAME=cityscapes_iouconf_hce2_1_concatInput_concatNet_dl #_more #_newspade
#IOUNET_NAME=cityscapes_conf_hce2_1_concatInput_concatNet_dl #_more #_newspade
IOUNET_NAME=cityscapes_iou_hce2_1_concatInput_concatNet_dl #_more #_newspade
#IOUNET_NAME=cityscapes_iouconf_hce2_1_baseline_dl #_newspade
#ln -s `pwd`/cityscapes/leftImg8bit $EXP_NAME/cityscapes/
#ln -s `pwd`/cityscapes/gtFine $EXP_NAME/cityscapes/
#IOUNET_NAME=cityscapes_iouconf_hce2_1_concatInput_concatNet_dl_reproduce
#IOUNET_NAME=cityscapes_iouconf_hce2_1_baseline_dl
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
#wait
#bash eval_deeplab_single.sh 2 0 $EXP_NAME &
#bash eval_deeplab_single.sh 3 1 $EXP_NAME &
#wait
#bash eval_deeplab_single_test.sh 2 $EXP_NAME

#bash eval_spade.sh 2 $EXP_NAME 200 train  &
#bash eval_spade.sh 3 $EXP_NAME 200 test &
#wait
#bash train_iounet.sh 3 $EXP_NAME $IOUNET_NAME $REC_PATH
bash eval_iounet.sh 2 $EXP_NAME $IOUNET_NAME $REC_PATH


