#-*- coding: utf-8 -*-

from src.human_detection import HumanDetection as ImgProcessor
import cv2
from config.config import video_path, write_box, write_video, frame_size, write_kps, gray
from utils.utils import boxdict2str, kpsdict2str, kpsScoredict2str
import copy
import numpy as np
import math


body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}
nece_point = [11, 13, 15]


IP = ImgProcessor()
fourcc = cv2.VideoWriter_fourcc(*'XVID')


def get_angle(center_coor, coor2, coor3):
    L1 = cal_dis(coor2, coor3)
    L2 = cal_dis(center_coor, coor3)
    L3 = cal_dis(center_coor, coor2)
    Angle = cal_angle(L1, L2, L3)
    return Angle


def cal_dis(coor1, coor2):
    out = np.square(coor1[0] - coor2[0]) + np.square(coor1[1] - coor2[1])
    return np.sqrt(out)


def cal_angle(L1, L2, L3):
    out = (np.square(L2) + np.square(L3) - np.square(L1)) / (2 * L2 * L3)
    try:
        return math.acos(out) * (180 / math.pi)
    except ValueError:
        return 180


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
                                  int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if write_video:
            try:
                self.out = cv2.VideoWriter(video_path[:-4] + "_processed.avi", fourcc, 15, (self.width, self.height))
                self.res_out = cv2.VideoWriter(video_path[:-4] + "_processed_res.avi", fourcc, 15, (self.width, self.height))
            except:
                self.out = cv2.VideoWriter("output.avi", fourcc, 15, (self.width, self.height))
                self.res_out = cv2.VideoWriter("output_res.avi", fourcc, 15, (self.width, self.height))
        if write_box:
            box_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_box.txt"
            self.box_txt = open(box_file, "w")
        if write_kps:
            kps_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps.txt"
            kps_score_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps_score.txt"
            self.kps_txt = open(kps_file, "w")
            self.kps_score_txt = open(kps_score_file, "w")

    def process_video(self):
        cnt = 0
        count = 0
        up_flag = 0
        count_squat = 0
        count_up = 0
        while True:
            ret, frame = self.cap.read()
            frame_save = copy.deepcopy(frame)
            cnt += 1
            if ret:
                # frame = cv2.resize(frame, frame_size)
                kps, boxes, kps_score = IP.process_img(frame, gray=gray)
                img, img_black = IP.visualize(h=self.height, w=self.width)
                preds = IP.classify(img_black)
                for location, pred in preds.items():
                # pred = IP.classify_whole(img_black)
                    cv2.putText(img, pred, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 3)

                if boxes is not None:
                    if write_box:
                        box_str = ""
                        for k, v in boxes.items():
                            box_str += boxdict2str(k, v)
                        self.box_txt.write(box_str)
                        self.box_txt.write("\n")
                else:
                    if write_box:
                        self.box_txt.write("\n")

                if kps:
                    # cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)
                    if write_kps:
                        kps_str = ""
                        for k, v in kps.items():
                            kps_str += kpsdict2str(k, v)
                        self.kps_txt.write(kps_str)
                        self.kps_txt.write("\n")

                        kps_score_str = ""
                        for k, v in kps_score.items():
                            kps_score_str += kpsScoredict2str(k, v)
                        self.kps_score_txt.write(kps_score_str)
                        self.kps_score_txt.write("\n")

                else:
                    if write_kps:
                        self.kps_txt.write("\n")
                        self.kps_score_txt.write("\n")
                    img = frame
                    # cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)
                try:
                    key_point = kps[1]
                    if len(img) > 0 and len(key_point) > 0:
                        coord = [key_point[idx] for idx in nece_point]
                        angle = get_angle(coord[1], coord[0], coord[2])
                        if angle > 60:
                            count_squat = 0 if count_squat == 0 else count_squat - 1
                            if count_up < 5:
                                count_up += 1
                            else:
                                count_up = 0
                                up_flag = 1
                        else:
                            if up_flag == 1:
                                count_up = 0 if count_up == 0 else count_up - 1
                                if count_squat > 4:
                                    count += 1
                                    up_flag = 0
                                    count_squat = 0
                                else:
                                    count_squat += 1
                            else:
                                pass
                except:
                    pass


                cv2.putText(img, "Count: {}".format(count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                cv2.imshow("res", cv2.resize(img, frame_size))
                cv2.imshow("res_black", cv2.resize(img_black, frame_size))
                cv2.waitKey(1)
                if write_video:
                    self.out.write(frame_save)
                    try:
                        self.res_out.write(img)
                    except:
                        pass
            else:
                self.cap.release()
                if write_video:
                    self.out.release()
                break

    def locate(self, kps):
        return kps


if __name__ == '__main__':
    # src_folder = "video/push_up"
    # dest_fodler = src_folder + "kps"
    # sub_folder = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder)]
    # sub_dest_folder = [os.path.join(dest_fodler, folder) for folder in os.listdir(src_folder)]

    VideoProcessor(video_path).process_video()
