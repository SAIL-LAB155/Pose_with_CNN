#-*- coding: utf-8 -*-

from src.human_detection import HumanDetection as ImgProcessor
import cv2
try:
    import src.debug.config.cfg_with_CNN as config
except:
    import config.config as config
from utils.utils import boxdict2str, kpsdict2str, kpsScoredict2str
import copy

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}

resize_ratio = config.resize_ratio
show_size = config.show_size
store_size = config.store_size
classify_type = config.classify_type
gray = config.gray
fourcc = cv2.VideoWriter_fourcc(*'XVID')


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = ImgProcessor(self.resize_size)

        if config.write_video:
            try:
                self.out = cv2.VideoWriter(video_path[:-4] + "_processed.avi", fourcc, 15, (self.width, self.height))
                self.res_out = cv2.VideoWriter(video_path[:-4] + "_processed_res.avi", fourcc, 15, (self.width, self.height))
            except:
                self.out = cv2.VideoWriter("output.avi", fourcc, 15, (self.width, self.height))
                self.res_out = cv2.VideoWriter("output_res.avi", fourcc, 15, (self.width, self.height))
        if config.write_box:
            box_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_box.txt"
            self.box_txt = open(box_file, "w")
        if config.write_kps:
            kps_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps.txt"
            kps_score_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps_score.txt"
            self.kps_txt = open(kps_file, "w")
            self.kps_score_txt = open(kps_score_file, "w")

    def process_video(self):
        self.IP.init()
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, self.resize_size)
                frame2 = copy.deepcopy(frame)
                kps, boxes, kps_score = self.IP.process_img(frame, gray=gray)
                img, img_black = self.IP.visualize()
                if classify_type == 1:
                    result = self.IP.classify_whole(img_black, img)
                elif classify_type == 2:
                    result = self.IP.classify_whole(frame2, img)
                elif classify_type == 3:
                    result = self.IP.classify(img_black, img, boxes)
                elif classify_type == 4:
                    result = self.IP.classify(frame2, img, boxes)
                else:
                    raise ValueError("Not a right classification type!")

                if boxes is not None:
                    if config.write_box:
                        box_str = ""
                        for k, v in boxes.items():
                            box_str += boxdict2str(k, v)
                        self.box_txt.write(box_str)
                        self.box_txt.write("\n")
                else:
                    if config.write_box:
                        self.box_txt.write("\n")

                if kps:
                    # cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)
                    if config.write_kps:
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
                    if config.write_kps:
                        self.kps_txt.write("\n")
                        self.kps_score_txt.write("\n")
                    img = frame
                    # cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)

                cv2.imshow("res", cv2.resize(img, show_size))
                cv2.waitKey(1)

                if config.write_video:
                    self.out.write(cv2.resize(frame, store_size))

            else:
                self.cap.release()
                if config.write_video:
                    self.out.release()
                break


if __name__ == '__main__':
    VideoProcessor(config.video_path).process_video()
