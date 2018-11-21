'''
    06/05/2018: The fast feature reader class for event detection on arbitrary length proposals.
    Running on AWS-1
'''

import os
import sys
sys.path.append("../")
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np


#read object from dict
def read_dict_from_pkl(fn):
    with open(fn,"rb") as f:
        obj=pickle.load(f)
    return obj

class FastFeatureReader(object):

    #   TODO: add multi-threading support for efficient IO?
    def __prefetch_feat_list__(self,feat_list):
        ret_dict=dict()

        #   the feature list should be similar to the instance list format.
        #   That is the first element in each meta is the proposal name (vid_name+"_"+pid)
        for feat_meta in feat_list:
            props_name=feat_meta[0]
            props_type=feat_meta[1]

            feat_fn=os.path.join(self.feat_root_path,props_type,self.feat_type,props_name+".npz")
            print "Prefetching: ",feat_fn
            if not os.path.isfile(feat_fn):
                continue

            feat=np.load(feat_fn)['feat']
            ret_dict[props_type+"/"+props_name]=feat

        print "Pre-fetching finished..."
        return ret_dict

    #   initializing without pre-fetching
    def __init__(self,feat_root_path,feat_type,prefetch_list=None):
        self.feat_type=feat_type
        self.feat_root_path=feat_root_path
        if prefetch_list is None:
            self.prefetch=False
            return

        #   prefetch all the features
        self.prefetch=True
        self.pid_feat_dict=self.__prefetch_feat_list__(prefetch_list)

    #   get parent feature
    def __get_par_props_feat__(self,props_type,par_props_id):
        par_full_id=props_type+"/"+par_props_id

        if self.prefetch and par_full_id in self.pid_feat_dict:
            return self.pid_feat_dict[par_full_id]

        feat_fn = os.path.join(self.feat_root_path, props_type, self.feat_type,par_props_id + ".npz")
        if not os.path.isfile(feat_fn):
            print feat_fn
            return None

        feat=np.load(feat_fn)['feat']
        return feat

    #   crop par feat (we assume that the first feat dim is the temporal feat)
    def __crop_par_feat__(self,par_feat,st_offset,end_offset):
        t_len=len(par_feat)
        if t_len==1:
            return par_feat
        t_st=int(t_len*st_offset);t_end=int(t_len*end_offset)

        if t_end==t_st:
            t_end+=1

        crop_feat=par_feat[t_st:t_end,::]
        return crop_feat

    #   load feature
    #   TODO: add support for recursive obtaining parent features?
    def load_feat(self,props_id,props_type,par_props_id,st_offset,end_offset):
        assert 0<=st_offset<=end_offset<=1
        par_feat=self.__get_par_props_feat__(props_type,par_props_id)
        if par_feat is None:
            print "[WARNING]: parent feature not exist for ",props_id,", type: ",props_type," id: ",par_props_id,"..."
            return None

        return self.__crop_par_feat__(par_feat,st_offset,end_offset)


if __name__=="__main__":
    num_test_ins=100
    #   test case with pre-defined instance list
    ins_list_fn="/home/jiangl1/data/datasets/diva_virat_data_v1_clean/instance_list/" \
                "export_gt_cls_baseline.new_negative.lwh_aug/split1/train.pkl"
    feat_root_path="/home/jiangl1/data/datasets/diva_virat_data_v1_clean/features"
    feat_type="i3d_rgb"

    print ins_list_fn
    ins_list=read_dict_from_pkl(ins_list_fn)

    #   Test with pre-fetching
    #feat_reader=FastFeatureReader(feat_root_path=feat_root_path,feat_type=feat_type,prefetch_list=ins_list)

    #   Test without pre-fetching
    feat_reader=FastFeatureReader(feat_root_path=feat_root_path,feat_type=feat_type,prefetch_list=None)

    #   Test case 1: obtain original features and cropped features
    for i in xrange(num_test_ins):
        cur_meta=ins_list[i]
        props_id=cur_meta[0];props_type=cur_meta[1];par_props_id=props_id;st_offset=0.0;end_offset=1.0
        print feat_reader.load_feat(props_id,props_type,par_props_id,st_offset,end_offset).shape

        st_offset=0.2;end_offset=0.8
        print "0.2 to 0.8: ",feat_reader.load_feat(props_id, props_type, par_props_id, st_offset, end_offset).shape


    print "done."