'''
    Pred the noun and verb classifier.
'''

import os
import sys
sys.path.append("../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
from data_loader.GenericDataLoader import DataLoader
from data_loader.NounVerbLoader import NounVerbLoader
import models.BasicNounVerbNN as basic_obj_nn
import numpy as np
from torch import nn,optim
#token_type="noun"
token_type="verb"

K=10

assert token_type in ["noun","verb"]

if token_type=="noun":
    token2id_dict = gen_utils.read_dict_from_pkl("/mnt/hdd_8t/TGIF/noun2id.pkl")
    id2token_dict = gen_utils.read_dict_from_pkl("/mnt/hdd_8t/TGIF/id2noun.pkl")
    model_root_path="/mnt/hdd_8t/TGIF/models/noun_det/"
    top_token_root_path=os.path.join(data_cfg.feat_root_path,"top_nouns")
    token_prob_root_path=os.path.join(data_cfg.feat_root_path,"noun_prob")

elif token_type=="verb":
    token2id_dict = gen_utils.read_dict_from_pkl("/mnt/hdd_8t/TGIF/verb2id.pkl")
    id2token_dict = gen_utils.read_dict_from_pkl("/mnt/hdd_8t/TGIF/id2verb.pkl")
    model_root_path = "/mnt/hdd_8t/TGIF/models/verb_det/"
    top_token_root_path = os.path.join(data_cfg.feat_root_path, "top_verbs")
    token_prob_root_path = os.path.join(data_cfg.feat_root_path, "verb_prob")

else:
    raise NotImplementedError

#   prepare the data loader
'''
dataloader = DataLoader(img_list=gen_utils.read_dict_from_pkl(data_cfg.all_list_fn),batch_size=32,
                        handler_obj=NounVerbLoader(feat_root_path="/mnt/hdd_8t/TGIF/features/jiac_i3d.motion_rgb.obj_det.place365_feat.resnet50",
                                                   text_root_path="/home/wcdu/11731_proj/gt_parsed/",
                                                   token2id_dict=token2id_dict,token_type=token_type))
'''

if not os.path.exists(top_token_root_path):
    os.makedirs(top_token_root_path)

if not os.path.exists(token_prob_root_path):
    os.makedirs(token_prob_root_path)

dataloader = NounVerbLoader(feat_root_path="/mnt/hdd_8t/TGIF/features/jiac_i3d.motion_rgb.obj_det.place365_feat.resnet50",
                        text_root_path="/home/wcdu/11731_proj/gt_parsed/",
                        token2id_dict=token2id_dict, token_type=token_type)

MODEL_FN=os.path.join(model_root_path,"model.pth")
assert os.path.isfile(MODEL_FN)

#   define the model
model=basic_obj_nn.BasicMultiClassNN(input_layer=7771,num_class=len(token2id_dict))
model=model.cuda()
model=gen_utils.load_model_by_state_dict(model,MODEL_FN)
model.eval()
all_list=gen_utils.read_dict_from_pkl(data_cfg.all_list_fn)

if __name__=="__main__":
    for cur_id in all_list:

        cur_batch=dataloader.get_internal_data_batch([cur_id])
        if len(cur_batch['feat'])==0:
            continue
        cur_feat=cur_batch['feat']
        cur_pred=model.forward(cur_feat)

        cur_pred=cur_pred.detach().cpu().numpy()
        cur_pred=np.squeeze(cur_pred)
        feat_fn=os.path.join(token_prob_root_path,str(cur_id)+".npz")
        np.savez_compressed(feat_fn,feat=cur_pred)

        max_prob=np.max(cur_pred)
        top_k_ind=cur_pred.argsort()[-K:][::-1].tolist()
        top_k_token=[id2token_dict[x] for x in top_k_ind]
        top_token_fn=os.path.join(top_token_root_path,str(cur_id)+".pkl")
        gen_utils.write_dict_to_pkl(top_k_token,top_token_fn)

        print "Extracting id: ", cur_id, "..."

    print "done."