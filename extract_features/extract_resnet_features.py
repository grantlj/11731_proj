import os
import sys
sys.path.append("../")
import utils.gen_utils as gen_utils
import torchvision.models as models
from data_loader.GenericDataLoader import *
from data_loader.SingleVideoPyTorchHandler import *
from torch import nn
import gc
import config.coco_dataset_config as data_cfg

print data_cfg.video_root_path
video_list_fn=data_cfg.all_list_fn
video_list=gen_utils.read_dict_from_pkl(video_list_fn)

model = models.resnet50(pretrained=True)
modules = list(model.children())[:-1]      # delete the last fc layer.
model = nn.Sequential(*modules)
model = model.cuda()
model.eval()

feat_type="resnet50"
feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

RE_EXTRACT_ALL=True

MAX_CHUNK_SIZE=5


def filter_video_list_by_existing_feat(video_list):
    ret_list=[]

    for id in video_list:
        dst_fn=os.path.join(feat_root_path,str(id)+".npz")
        if not os.path.isfile(dst_fn) or RE_EXTRACT_ALL:
            ret_list.append(id)
    return ret_list

if  __name__=="__main__":


    video_list=filter_video_list_by_existing_feat(video_list)
    #video_list = filter_video_list_by_existing_feat([9299])
    handler = SingleVideoHandler(key_frame_interval=1, video_root_path=data_cfg.video_root_path)
    for id in video_list:
        cur_batch=handler.get_internal_data_batch([id])
        video_tensors, vid_ids = cur_batch['video'], cur_batch['id_list']
        for vid_tensor, vid_id in zip(video_tensors, vid_ids):

            dst_fn = os.path.join(feat_root_path, str(vid_id) + ".npz")
            vid_tensor = vid_tensor.type('torch.FloatTensor').cuda()

            vid_tensor_list = torch.split(vid_tensor, MAX_CHUNK_SIZE, dim=0)

            vid_tensor = None
            vid_feat = None
            for vid_tensor_chunk in vid_tensor_list:

                vid_chunk_feat = model.forward(vid_tensor_chunk)
                vid_chunk_feat = vid_chunk_feat.detach().cpu().numpy()
                vid_chunk_feat = np.squeeze(vid_chunk_feat, axis=2)
                vid_chunk_feat = np.squeeze(vid_chunk_feat, axis=2)

                if vid_feat is None:
                    vid_feat = vid_chunk_feat
                else:
                    vid_feat = np.concatenate((vid_feat, vid_chunk_feat), axis=0)

            np.savez_compressed(dst_fn, feat=vid_feat)
            vid_feat = None
            print "Extracting vid id: ", vid_id

        gc.collect()

'''
    dataloader = DataLoader(img_list=video_list, batch_size=1,
                            handler_obj=SingleVideoHandler(key_frame_interval=8,
                                                           video_root_path="/home/jiangl1/data/datasets/TGIF/videos/"),num_worker=1)
    dataloader.reset_reader()
    dataloader.start_producer_queue()
    
    while True:
        cur_batch=dataloader.get_data_batch()
        if cur_batch is None:
            break

        video_tensors,vid_ids=cur_batch['video'],cur_batch['id_list']
        for vid_tensor,vid_id in zip(video_tensors,vid_ids):

            dst_fn=os.path.join(feat_root_path,str(vid_id)+".npz")
            vid_tensor=vid_tensor.type('torch.FloatTensor').cuda()

            vid_tensor_list=torch.split(vid_tensor,MAX_CHUNK_SIZE,dim=0)

            vid_tensor=None
            vid_feat=None
            for vid_tensor_chunk in vid_tensor_list:

                vid_chunk_feat=model.forward(vid_tensor_chunk)
                vid_chunk_feat=vid_chunk_feat.detach().cpu().numpy()
                vid_chunk_feat=np.squeeze(vid_chunk_feat,axis=2)
                vid_chunk_feat=np.squeeze(vid_chunk_feat,axis=2)

                if vid_feat is None:
                    vid_feat=vid_chunk_feat
                else:
                    vid_feat=np.concatenate((vid_feat,vid_chunk_feat),axis=0)

            np.savez_compressed(dst_fn,feat=vid_feat)
            vid_feat=None
            print "Extracting vid id: ", vid_id

        gc.collect()
'''