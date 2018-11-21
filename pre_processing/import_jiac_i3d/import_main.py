import os
import sys
sys.path.append("../")
sys.path.append("../../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
import numpy as np

dst_feat_root_path=os.path.join(data_cfg.feat_root_path,"jiac_i3d")
if not os.path.exists(dst_feat_root_path):
    os.makedirs(dst_feat_root_path)


jia_root_path="/home/jiangl1/data/datasets/TGIF/imports/"

trn_vid_fn=os.path.join(jia_root_path,"jiac_videoids","trn_videoids.npy")
val_vid_fn=os.path.join(jia_root_path,"jiac_videoids","val_videoids.npy")
tst_vid_fn=os.path.join(jia_root_path,"jiac_videoids","tst_videoids.npy")
int2vid_fn=os.path.join(jia_root_path,"jiac_videoids","int2video.npy")

jiac_trn_ids=np.load(trn_vid_fn)
jiac_val_ids=np.load(val_vid_fn)
jiac_tst_ids=np.load(tst_vid_fn)
jiac_int2video=np.load(int2vid_fn)
jiac_int2vid_dict={}

vid2lj_id_dict={}

for id in xrange(0,len(jiac_int2video)):
    jiac_int2vid_dict[id]=jiac_int2video[id]

#   get video name to lj_id mapping
org_data_fn="/home/jiangl1/data/datasets/TGIF/TGIF-Release/data/tgif-v1.0.tsv"
org_data_lines=gen_utils.read_lines_from_text_file(org_data_fn)
lid=0

for now_line in org_data_lines:
    tmp_str=now_line.split("\t")[0]
    tumbler_id=tmp_str.split("/")[-1].replace(".gif","")
    vid2lj_id_dict[tumbler_id]=lid
    lid+=1
    pass

print "initialize finished,  jiac int2vid size:",len(jiac_int2video)
print "done."

def handle_split(jiac_ids,feat_fn):
    all_feats=np.load(feat_fn)
    feat_id=0
    for jia_id in jiac_ids:
        feat=all_feats[feat_id]
        feat_id+=1

        lj_id=vid2lj_id_dict[jiac_int2video[jia_id]]
        print "lj_id",lj_id

        dst_feat_fn=os.path.join(dst_feat_root_path,str(lj_id)+".npz")
        np.savez_compressed(dst_feat_fn,feat=feat)
        pass

    return

if __name__=="__main__":
    handle_split(jiac_trn_ids,os.path.join(jia_root_path,"jiaci3d","trn_ft.npy"))
    handle_split(jiac_val_ids,os.path.join(jia_root_path,"jiaci3d","val_ft.npy"))
    handle_split(jiac_tst_ids,os.path.join(jia_root_path,"jiaci3d","tst_ft.npy"))
    print "done."