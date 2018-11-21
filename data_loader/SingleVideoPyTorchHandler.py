import os
import sys
sys.path.append("../")
import cv2
import numpy as np
import torch
import utils.gen_utils as gen_utils
from GenericDataLoader import DataLoader
import gc

'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
'''

mean=np.asarray([0.485, 0.456, 0.406])
std=np.asarray([0.229, 0.224, 0.225])

class SingleVideoHandler(object):
    def __init__(self,video_root_path,key_frame_interval=1,is_train=False):
        self.key_frame_interval=key_frame_interval
        self.video_root_path=video_root_path

    def trans_into_tensor_for_feat_ext(self,vid_frames):
        n_fr=len(vid_frames)
        for i in xrange(0,n_fr):
            cur_fr=vid_frames[i]
            cur_fr=cv2.resize(cur_fr,(224,224))
            cur_fr=cur_fr[:,:,::-1]
            cur_fr=cur_fr/float(255)
            cur_fr = (cur_fr - mean) / std
            vid_frames[i]=cur_fr

            #vid_frames[i]=normalize(vid_frames[i])

        vid_frames=np.asarray(vid_frames)
        vid_frames=np.transpose(vid_frames,axes=(0,3,1,2))
        vid_frames=torch.from_numpy(vid_frames)
        gc.collect()
        return vid_frames

    def sample_list_with_interval(self,vid_frames):
        ret_list=[]
        for i in xrange(0,len(vid_frames),self.key_frame_interval):
            ret_list.append(vid_frames[i])
        vid_frames=None
        return ret_list

    def get_internal_data_batch(self,batch_id_list):
        ret_vid_list=[];ret_id_list=[]
        for vid_id in batch_id_list:
            vid_fn=os.path.join(self.video_root_path,str(vid_id)+".mp4")
            if not os.path.isfile(vid_fn):
                continue
            ret_id_list.append(vid_id)

            try:
                vid_frames=gen_utils.read_frame_list_from_video_file_sk(vid_fn,verbose=False,max_fr=3000)
            except:
                vid_frames=None
                print "video reading failed:",vid_fn
                print "failed..."

            vid_frames=self.sample_list_with_interval(vid_frames)

            vid_frames=self.trans_into_tensor_for_feat_ext(vid_frames)

            ret_vid_list.append(vid_frames)

        #   a list of video tensors
        return {'video':ret_vid_list,'id_list':ret_id_list}


#   the tester for generic data loader

if __name__=="__main__":
    import config.dataset_config as data_cfg
    dataloader=DataLoader(lst_fn=data_cfg.trn_list_fn,batch_size=4,
                          handler_obj=SingleVideoHandler(key_frame_interval=10,video_root_path="/home/jiangl1/data/datasets/TGIF/videos/"))
    dataloader.shuffle_data()
    dataloader.reset_reader()
    dataloader.start_producer_queue()
    step=0
    while True:
        step+=1
        batch=dataloader.get_data_batch()
        if batch is None:
            break
        id_list=batch['id_list']
        print id_list,step

    print "done."
'''
if __name__=="__main__":
    handler=SingleVideoHandler(key_frame_interval=10,video_root_path="/home/jiangl1/data/datasets/TGIF/videos/")
    handler.get_internal_data_batch([1,2,3])
    print "done."
'''