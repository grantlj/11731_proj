import os
import sys
sys.path.append("../")
import cv2
import numpy as np
import torch
import utils.gen_utils as gen_utils
from GenericDataLoader import DataLoader
import gc
import pickle
import pdb
from string import punctuation

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


class VideoTextHandler(SingleVideoHandler):
    def __init__(self, video_root_path, text_root_path, obj_det_path, mot_rec_path, place_path, frame_path, dic, key_frame_interval=1, is_train=False):
        self.key_frame_interval = key_frame_interval
        self.video_root_path = video_root_path
        self.text_root_path = text_root_path
        self.obj_det_path = obj_det_path
        self.mot_rec_path = mot_rec_path
        self.place_path = place_path
        self.frame_path = frame_path
        self.dictionary = dic
        self.start = dic['<start>']
        self.eou = dic['<eou>']

    def get_internal_data_batch(self,batch_id_list):
        ret_vid_list = []
        ret_txt_list = []
        ret_id_list = []
        ret_nnid_list = []
        ret_vbid_list = []
        ret_tag_list = []
        ret_obj_list = []
        ret_mot_list = []
        ret_frame_list = []
        ret_place_list = []
        for vid_id in batch_id_list:
            vid_fn = os.path.join(self.video_root_path,str(vid_id)+".npz")
            mot_fn = os.path.join(self.mot_rec_path , str(vid_id) + '.npz')
            frame_fn = os.path.join(self.frame_path , str(vid_id) + '.npz')
            place_fn = os.path.join(self.place_path , str(vid_id) + '.npz')
            if not os.path.isfile(vid_fn) or \
               not os.path.isfile(mot_fn) or \
               not os.path.isfile(frame_fn) or \
               not os.path.isfile(place_fn):
                continue

            vid_frames = np.load(vid_fn)['feat']
            ret_id_list.append(vid_id)

            '''
            obj_fn = os.path.join(self.obj_det_path , str(vid_id) + '.pkl')
            obj_det = pickle.load(open(obj_fn, 'rb'))
            objects = [x['cat'] for d in obj_det for x in obj_det[d] if x['score'] > 0.8]
            objects = list(set(objects))
            ret_obj_list.append(objects)
            '''

            mot_rec = np.load(mot_fn)['feat']
            ret_mot_list.append(mot_rec)

            text_fn = os.path.join(self.text_root_path, str(vid_id) + '.pkl')
            data = pickle.load(open(text_fn, 'rb'))
        
            text = data['txt']
            text = [self.start] + [self.dictionary[w] for w in text] + [self.eou]
            ret_txt_list.append(text)
            nn_id = [x + 1 for x in data['nn_id']]
            vb_id = [x + 1 for x in data['vb_id']]
            ret_nnid_list.append(nn_id)
            ret_vbid_list.append(vb_id)
            tags = data['tags']
            ret_tag_list.append(tags)
            
            ret_vid_list.append(vid_frames)

            frames = np.load(frame_fn)['feat']
            ret_frame_list.append(frames)

            places = np.load(place_fn)['feat']
            ret_place_list.append(places)

        #   a list of video tensors
        return {'video':ret_vid_list,'id_list':ret_id_list, 'text': ret_txt_list, 
                'nn_id': ret_nnid_list, 
                'vb_id': ret_vbid_list, 
                'mot': ret_mot_list,
                'frame': ret_frame_list,
                'place': ret_place_list,
                'tags': ret_tag_list, 'obj': ret_obj_list}


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
