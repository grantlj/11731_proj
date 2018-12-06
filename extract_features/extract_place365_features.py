import os
import sys
sys.path.insert(0,"../")

#   for TGIF
import config.dataset_config as data_cfg

#   for COCO
#import config.coco_dataset_config as data_cfg
import utils.gen_utils as gen_utils
import numpy as np
import config.place365_config as place_cfg
import cv2
from torchvision import transforms, utils
import PIL.Image as Image
from torch.autograd import Variable
import torch
import threading
import torchvision.models as models
from torch import nn

video_list_fn=data_cfg.all_list_fn
video_list=gen_utils.read_dict_from_pkl(video_list_fn)

prob_feat_root_path=os.path.join(data_cfg.feat_root_path,"place365_prob_dist")
feat_root_path=os.path.join(data_cfg.feat_root_path,"place365_feat")

if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

if not os.path.exists(prob_feat_root_path):
    os.makedirs(prob_feat_root_path)

RE_EXTRACT=False
key_frame_interval=8

# load the image transformer
normalize = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#   load the model
model_file="/mnt/hdd_8t/place365/resnet18_places365.pth.tar"
model = models.__dict__['resnet18'](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.cuda()
model.eval()

feat_model=nn.Sequential(*list(model.children())[:-1]).cuda()
feat_model.eval()

gpu_id_list=[4,5]
print "Initialize finished..."

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
        vid_logits_fn = os.path.join(prob_feat_root_path, str(vid) + ".npz")
        vid_feat_fn=os.path.join(feat_root_path,str(vid)+".npz")

        if os.path.isfile(vid_logits_fn) and os.path.isfile(vid_feat_fn) and not RE_EXTRACT:
            continue

        if not os.path.isfile(vid_fn):
            continue

        vid_frames = gen_utils.read_frame_list_from_video_file(vid_fn, verbose=True, max_fr=7000)
        vid_frames = sample_list_with_interval(vid_frames, key_frame_interval=key_frame_interval)

        feat_mat=None
        fr_feat_mat=None
        for fr in vid_frames:
            fr=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr=Image.fromarray(np.uint8(fr))
            fr_tensor=normalize(fr)
            fr_tensor=torch.unsqueeze(fr_tensor,dim=0)
            fr_tensor=Variable(fr_tensor.type(torch.FloatTensor), volatile=True).cuda()

            fr_logits=model.forward(fr_tensor)
            fr_logits=fr_logits.detach().cpu().numpy()

            fr_feat=feat_model.forward(fr_tensor)
            fr_feat=fr_feat.detach().cpu().numpy()
            fr_feat=np.squeeze(fr_feat)
            fr_feat=np.expand_dims(fr_feat,axis=0)

            if feat_mat is None:
                feat_mat=fr_logits
                fr_feat_mat=fr_feat
            else:
                feat_mat=np.concatenate((feat_mat,fr_logits),axis=0)
                fr_feat_mat=np.concatenate((fr_feat_mat,fr_feat),axis=0)

        print vid_feat_fn
        np.savez_compressed(vid_logits_fn,feat=feat_mat)
        np.savez_compressed(vid_feat_fn,feat=fr_feat_mat)

    return

if __name__=="__main__":
    num_gpu=len(gpu_id_list)
    sub_vid_lists=gen_utils.split_list_into_n_folder_parts(video_list,num_gpu)

    thread_pool=[]
    for vid_list,gpu_id in zip(sub_vid_lists,gpu_id_list):
        th=threading.Thread(target=handle_a_particular_list,args=(vid_list,gpu_id,))
        th.start()
        thread_pool.append(th)


    for th in thread_pool:
        th.join()

    print "done."