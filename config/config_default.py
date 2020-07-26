import torch

video_path = "video/ceiling/frog_1.mp4"
img_folder = "video/ceiling/BU"

write_video = False
write_box = False
write_kps = False

device = "cuda:0"
print("Using {}".format(device))

confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64

fast_inference = True
pose_batch = 80


frame_size = (720, 540)

pose_backbone = "seresnet101"
pose_weight = "weights/sppe/duc_se.pth"
pose_cfg = None
pose_cls = 17

DUCs = [480, 240]

CNN_weight = "weights/CNN/ceiling/ceiling_mobilenet_2cls.pth"
CNN_backbone = "mobilenet"
CNN_class = ["freestyle", "frog"]


yolo_cfg = "config/yolo_cfg/yolov3-1cls.cfg"
yolo_weight = 'weights/yolo/0607_bs32_freeT0.8.weights'


track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True
