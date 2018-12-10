'''
    Feature concatenation.
'''

import os
import sys
sys.path.append("../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
import numpy as np
import threading

todo_feat_name_list=[
                     ['jiac_i3d','resnet50','place365_feat','motion_rgb']]

feat_root_path=data_cfg.feat_root_path
lst=gen_utils.read_dict_from_pkl(data_cfg.all_list_fn)

obj_labels=gen_utils.read_dict_from_pkl("/home/jiangl1/data/datasets/TGIF/models/obj_det/faster_rcnn/id2label.pkl")
n_obj=len(obj_labels)
obj_label_list=[]
for obj_id in sorted(obj_labels.keys()):
    obj_label_list.append(obj_labels[obj_id])

print "Initialize finished..."

def load_obj_det_feat(cur_feat_fn):
    feat=gen_utils.read_dict_from_pkl(cur_feat_fn)

    all_fr_feat=[]
    for fr,obj_list in feat.iteritems():
        fr_feat=np.zeros(n_obj)
        for obj_meta in obj_list:
            obj_name=obj_meta['cat']
            obj_ind=obj_label_list.index(obj_name)
            fr_feat[obj_ind]=1.0
        all_fr_feat.append(fr_feat)

    #ret_feat=np.nanmean(all_fr_feat,axis=0)
    return np.asarray(all_fr_feat)

def get_feat_dict(cur_id,feat_name_list):
    ret_dict={}
    for feat_name in feat_name_list:
        print feat_name

        if feat_name=="obj_det":
            cur_feat_fn=os.path.join(feat_root_path,feat_name,str(cur_id)+".pkl")
        else:
            cur_feat_fn=os.path.join(feat_root_path,feat_name,str(cur_id)+".npz")

        if not os.path.isfile(cur_feat_fn):
            print "File not exist:",cur_feat_fn
            ret_dict=None
            break

        if feat_name=="obj_det":
            feat=load_obj_det_feat(cur_feat_fn)

            if len(feat.shape)==2:
                feat=np.nanmean(feat,axis=0)
            elif len(feat.shape)==1:
                #feat=np.expand_dims(feat,axis=0)
                pass

        else:
            feat=np.load(cur_feat_fn)['feat']
            if len(feat.shape) == 2:
                feat=np.nanmean(feat,axis=0)
            elif len(feat.shape)==1:
                #feat=np.expand_dims(feat,axis=0)
                pass

        ret_dict[feat_name]=feat

    return ret_dict

def handle_a_feat_name(feat_name_list):
    feat_name_list=sorted(feat_name_list)
    dst_feat_name=".".join(feat_name_list)
    dst_feat_root_path=os.path.join(feat_root_path,dst_feat_name)
    if not os.path.exists(dst_feat_root_path):
        os.makedirs(dst_feat_root_path)

    for cur_id in lst:
        dst_fn=os.path.join(dst_feat_root_path,str(cur_id)+".npz")
        print "Handling cur_id: ",cur_id
        feat_dict=get_feat_dict(cur_id,feat_name_list)
        if feat_dict is None:
            continue

        final_feat=None
        for feat_name in feat_name_list:
            cur_feat=feat_dict[feat_name]
            if final_feat is None:
                final_feat=cur_feat
            else:
                final_feat=np.concatenate((final_feat,cur_feat),axis=0)

        np.savez_compressed(dst_fn,feat=final_feat)
    return

if __name__=="__main__":
    thread_pool=[]

    for feat_name_list in todo_feat_name_list:
        th=threading.Thread(target=handle_a_feat_name,args=(feat_name_list,))
        th.start()
        thread_pool.append(th)

        #handle_a_feat_name(feat_name_list)

    for th in thread_pool:
        th.join()

    print "done."