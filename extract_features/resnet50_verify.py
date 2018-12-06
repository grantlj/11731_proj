import os
import sys
sys.path.append("../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
import numpy as np

feat_root_path="/home/jiangl1/data/datasets/TGIF/features/resnet50/"

lst=gen_utils.read_dict_from_pkl(data_cfg.all_list_fn)

if __name__=="__main__":
    for id in lst:
        if id%1000==0:
            print id
        feat_fn=os.path.join(feat_root_path,str(id)+".npz")
        if not os.path.isfile(feat_fn):
            #print "Not exist: ",id
            continue
        try:
            x=np.load(feat_fn)['feat']
        except:
            print "Error:",id
    print "done."