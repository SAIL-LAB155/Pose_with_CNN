import torch
# from config import config
from src.yolo.preprocess import prep_frame
from src.yolo.util import dynamic_write_results
from src.yolo.darknet import Darknet
import numpy as np
from src.utils.benchmark import print_model_param_flops, print_model_param_nums, get_inference_time
from src.opt import opt

input_size = opt.input_size
device = opt.device
confidence = opt.confidence
num_classes = opt.num_classes
nms_thresh = opt.nms_thresh

onnx = opt.onnx
libtorch = opt.libtorch

empty_tensor = torch.empty([0, 8])


class ObjectDetectionYolo(object):
    def __init__(self, cfg, weight, batchSize=1):
        self.det_model = Darknet(cfg)
        # self.det_model.load_state_dict(torch.load('models/yolo/yolov3-spp.weights', map_location="cuda:0")['model'])
        self.det_model.load_weights(weight)
        self.det_model.net_info['height'] = input_size
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        if device != "cpu":
            self.det_model.cuda()
        inf_time = get_inference_time(self.det_model, height=input_size, width=input_size)
        flops = print_model_param_flops(self.det_model, input_width=input_size, input_height=input_size)
        params = print_model_param_nums(self.det_model)
        print("Detection: Inference time {}s, Params {}, FLOPs {}".format(inf_time, params, flops))
        if libtorch:
            example = torch.rand(2, 3, 224, 224)
            traced_model = torch.jit.trace(self.det_model, example)
            traced_model.save("det_lib.pt")
        self.det_model.eval()
        self.im_dim_list = []
        self.batchSize = batchSize
        self.mul_img = False

    def __preprocess(self, frame):
        img = []
        orig_img = []
        im_dim_list = []
        if len(frame.shape) == 3:
            frame = np.expand_dims(frame, axis=0)
            self.mul_img = False
        else:
            self.mul_img = True

        for k in range(frame.shape[0]):
            img_k, orig_img_k, im_dim_list_k = prep_frame(frame[k], int(input_size))
            img.append(img_k)
            orig_img.append(orig_img_k)
            im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        return img, im_dim_list

    def __detect(self, img, im_dim_list):
        self.im_dim_list = im_dim_list
        with torch.no_grad():
            # Human Detection
            if device != "cpu":
                img = img.cuda()

            prediction = self.det_model(img)
            # NMS process
            dets = dynamic_write_results(prediction, confidence,  num_classes, nms=True, nms_conf=nms_thresh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                return empty_tensor

            dets = dets.cpu()
            self.im_dim_list = torch.index_select(self.im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / self.im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * self.im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * self.im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
        return dets

    def process(self, frame):
        img, im_dim_list = self.__preprocess(frame)
        det_res = self.__detect(img, im_dim_list)
        # boxes, scores = self.cut_box_score(det_res)
        # return boxes, scores
        if self.mul_img:
            return det_res
        return det_res[:,1:]

    def cut_box_score(self, results):
        if len(results) == 0:
            return empty_tensor, empty_tensor

        for j in range(results.shape[0]):
            results[j, [0, 2]] = torch.clamp(results[j, [0, 2]], 0.0, self.im_dim_list[j, 0])
            results[j, [1, 3]] = torch.clamp(results[j, [1, 3]], 0.0, self.im_dim_list[j, 1])
        boxes = results[:, 0:4]
        scores = results[:, 4:5]

        # boxes_k = boxes[results[:, 0] == 0]
        # if isinstance(boxes, int) or boxes.shape[0] == 0:
        #     return None, None

        return boxes, scores

