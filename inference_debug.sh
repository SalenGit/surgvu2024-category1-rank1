rm -r runs

CUDA_VISIBLE_DEVICES=1 python detect.py --source test/vid_1_short.mp4 --half --img 800 --conf-thres 0.6 --iou-thres 0.01 --weights stage_0_weights/full.pt --name "video_test" --save-txt --save-conf --nosave