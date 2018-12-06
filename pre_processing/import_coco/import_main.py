'''
    11/24/2018: Import the COCO dataset for training.
'''

import os
import sys
sys.path.append("../")
sys.path.append("../../")
import cv2
import config.coco_dataset_config as data_cfg
import utils.gen_utils as gen_utils

trn_img_root_path="/home/jiangl1/data/datasets/COCO/raw/train2014/"
val_img_root_path="/home/jiangl1/data/datasets/COCO/raw/val2014/"
trn_anno_fn="/home/jiangl1/data/datasets/COCO/raw/annotations/captions_train2014.json"
val_anno_fn="/home/jiangl1/data/datasets/COCO/raw/annotations/captions_val2014.json"

trn_anno=gen_utils.read_dict_from_json(trn_anno_fn)
val_anno=gen_utils.read_dict_from_json(val_anno_fn)

print "Initialize finished..."

def handle_a_split(img_root_path,anno,id_offset,prefix):

    id2anno_list={}
    for cur_anno in anno['annotations']:
        img_id=cur_anno['image_id']
        caption=cur_anno['caption']
        if not img_id in id2anno_list:
            id2anno_list[img_id]=[]
        id2anno_list[img_id].append(caption)

    id_list=[]
    for img_meta in anno['images']:
        id_offset+=1

        img_id=img_meta['id']
        print img_id,',',id_offset,"/",str(len(anno['images']))

        assert img_id in id2anno_list
        org_img_fn=os.path.join(img_root_path,"%s_%012d.jpg"%(prefix,img_id))
        dst_img_fn = os.path.join(data_cfg.video_root_path, str(id_offset) + ".mp4")
        id_list.append(id_offset)

        if os.path.isfile(dst_img_fn):
            continue

        if not os.path.isfile(org_img_fn):
            continue

        im=cv2.imread(org_img_fn)


        dst_anno_fn=os.path.join(data_cfg.gt_root_path,str(id_offset)+".pkl")
        cur_anno=id2anno_list[img_id]

        gen_utils.write_dict_to_pkl(cur_anno,dst_anno_fn)
        gen_utils.write_frame_list_to_video_file(dst_img_fn,[im])

    return id_offset,id_list

if __name__=="__main__":
    id_offset=0

    id_offset,trn_list=handle_a_split(trn_img_root_path,trn_anno,id_offset,"COCO_train2014")
    id_offset,val_list=handle_a_split(val_img_root_path,val_anno,id_offset,"COCO_val2014")

    all_list=trn_list+val_list

    gen_utils.write_dict_to_pkl(trn_list,data_cfg.trn_list_fn)
    gen_utils.write_dict_to_pkl(val_list,data_cfg.tst_list_fn)
    gen_utils.write_dict_to_pkl(val_list,data_cfg.val_list_fn)
    gen_utils.write_dict_to_pkl(all_list,data_cfg.all_list_fn)

    print "done."