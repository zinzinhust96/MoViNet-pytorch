CUDA_VISIBLE_DEVICE=1 python train.py \
--video_path data/video_data_1 \
--annotation_path data/test_train_splits_1 \
--num_frames 16 \
--clip_steps 2 \
--bs_train 16 \
--bs_test 16 \
--lr 0.00005 \
--gpu 1 \
--pretrained results/theft/model_epoch0005_loss1.0017_acc77.00.pth