import os
import sys
sys.path.append("../")
sys.path.append("../../")
import utils.gen_utils as gen_utils
import cv2

vid_fn="/mnt/hdd_8t/COCO/videos/9999.mp4"
gt_fn="/mnt/hdd_8t/COCO/gt/9999.pkl"

if __name__=="__main__":
    vid_frames=gen_utils.read_frame_list_from_video_file_sk(vid_fn)
    gt=gen_utils.read_dict_from_pkl(gt_fn)
    im=vid_frames[0]
    cv2.imwrite("/home/jiangl1/tmp.jpg",im)
    print gt
    print "done."