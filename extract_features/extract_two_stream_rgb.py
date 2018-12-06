import os
import sys
sys.path.insert(0,"../")

#   for TGIF
import config.dataset_config as data_cfg

#   for COCO
#import config.coco_dataset_config as data_cfg
import utils.gen_utils as gen_utils
import numpy as np
import cv2
from torchvision import transforms, utils
import PIL.Image as Image
from torch.autograd import Variable
import torch
import threading
import torchvision.models as models
from torch import nn
import twostream_rgb

video_list_fn=data_cfg.all_list_fn
video_list=gen_utils.read_dict_from_pkl(video_list_fn)

feat_root_path=os.path.join(data_cfg.feat_root_path,"motion_rgb")

if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

RE_EXTRACT=False
key_frame_interval=8

gpu_id_list=[0,1]

model=twostream_rgb.twostream_rgb
model.load_state_dict(torch.load("/mnt/hdd_8t/charades/models/two_stream_baseline/twostream_rgb.pth"))

for i in xrange(38,39):
    st_i=str(i)
    del model._modules[st_i]
model.cuda()

print "Initialize finished..."

def convert_np_chunk_to_tensor_rgb_with_org_scale(np_chunk, img_size, do_crop=False):
    pass

    color_normalize=transforms.Normalize(
        mean=[103.939/255, 116.779/255, 123.68/255],                               # R, G, B
        #mean=[123.68 / 255, 103.939 / 255, 116.779 / 255],                         # B, G, R
        std=[1.0,1.0,1.0]
    )
    if do_crop:
        preprocessor=transforms.Compose([
                                        transforms.Scale(img_size),
                                        transforms.ToTensor(),
                                        color_normalize,
                                        ])
    else:
        preprocessor = transforms.Compose([
            transforms.ToTensor(),
            color_normalize,
        ])

    cur_tensor=[]
    for img in np_chunk:
       pass
       img_tensor=preprocessor(img)
       img_tensor.unsqueeze_(0)
       if cur_tensor==[]:
           cur_tensor=img_tensor
       else:
           cur_tensor=torch.stack((cur_tensor,img_tensor))

    return cur_tensor

def sample_list_with_interval(vid_frames,key_frame_interval):
    ret_list=[]
    for i in xrange(0,len(vid_frames),key_frame_interval):
        ret_list.append(vid_frames[i])
    vid_frames=None
    return ret_list

def handle_a_particular_list(vid_list,gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for vid in vid_list:

        vid_fn = os.path.join(data_cfg.video_root_path, str(vid) + ".mp4")
        vid_feat_fn=os.path.join(feat_root_path,str(vid)+".npz")

        if os.path.isfile(vid_feat_fn) and not RE_EXTRACT:
            continue
        if not os.path.isfile(vid_fn):
            continue

        print "Extracting: ",vid_fn

        vid_frames = gen_utils.read_frame_list_from_video_file(vid_fn, verbose=True, max_fr=7000)
        vid_frames = sample_list_with_interval(vid_frames, key_frame_interval=key_frame_interval)

        feat_mat=None
        for fr in vid_frames:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr,(224,224))
            fr = Image.fromarray(np.uint8(fr))
            fr=[fr]
            cur_tensor = convert_np_chunk_to_tensor_rgb_with_org_scale(fr,224,do_crop=True)
            cur_tensor = Variable(cur_tensor.type(torch.FloatTensor).cuda())

            cur_feat=model.forward(cur_tensor)
            cur_feat=cur_feat.detach().cpu().numpy()

            if feat_mat is None:
                feat_mat=cur_feat
            else:
                feat_mat=np.concatenate((feat_mat,cur_feat),axis=0)

        np.savez_compressed(vid_feat_fn,feat=feat_mat)

    return

if __name__=="__main__":

    num_gpu = len(gpu_id_list)
    sub_vid_lists = gen_utils.split_list_into_n_folder_parts(video_list, num_gpu)

    thread_pool = []
    for vid_list, gpu_id in zip(sub_vid_lists, gpu_id_list):
        th = threading.Thread(target=handle_a_particular_list, args=(vid_list, gpu_id,))
        th.start()
        thread_pool.append(th)

    for th in thread_pool:
        th.join()

    print "done."