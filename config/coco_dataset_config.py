import os
import sys
sys.path.append("../")

#   on AWS-2
dataset_root_path="/mnt/hdd_8t/COCO/"

raw_data_root_path=os.path.join(dataset_root_path,"raw")

#   split root path
split_root_path=os.path.join(dataset_root_path,"splits")
if not os.path.exists(split_root_path):
    os.makedirs(split_root_path)

trn_list_fn=os.path.join(split_root_path,"trn.pkl")
val_list_fn=os.path.join(split_root_path,"val.pkl")
tst_list_fn=os.path.join(split_root_path,"tst.pkl")
all_list_fn=os.path.join(split_root_path,"all.pkl")

gt_root_path=os.path.join(dataset_root_path,"gt")
video_root_path=os.path.join(dataset_root_path,"videos")
feat_root_path=os.path.join(dataset_root_path,"features")

if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)