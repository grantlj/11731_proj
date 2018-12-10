import sys
sys.path.append("../")
import torch
import pickle
import config.dataset_config as data_cfg
import numpy as np
import argparse
from data_loader.GenericDataLoader import DataLoader
from data_loader.SingleVideoPyTorchHandler import VideoTextHandler
from model import Baseline
from torch.nn.utils import clip_grad_norm
import utils.gen_utils as gen_utils


parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str, default='baseline_template_lj')
config = parser.parse_args()
txt_root = '/home/wcdu/11731_proj/gt_parsed_nv_template/'
dictionary = pickle.load(open('/home/wcdu/11731_proj/template_pos_dict.pkl', 'rb'))
def get_loader(split):
    if split == 'train':
        cfg_list_fn = data_cfg.trn_list_fn
    elif split == 'valid':
        cfg_list_fn = data_cfg.val_list_fn
    else:
        raise NotImplementedError
    dataloader = DataLoader(img_list=pickle.load(open(cfg_list_fn, 'rb')),
                            batch_size=16,
                            num_worker=4,
                            handler_obj=VideoTextHandler(video_root_path="/mnt/hdd_8t/TGIF/features/jiac_i3d",
                                                         key_frame_interval=8,
                                                         text_root_path=txt_root,
                                                         obj_det_path='/mnt/hdd_8t/TGIF/features/obj_det',
                                                         mot_rec_path='/mnt/hdd_8t/TGIF/features/motion_rgb',
                                                         place_path='/mnt/hdd_8t/TGIF/features/place365_feat',
                                                         frame_path='/mnt/hdd_8t/TGIF/features/resnet50',
                                                         dic=dictionary['word2id']))
    return dataloader

model = torch.load('../{}.pt'.format(config.model)).cuda()
#params = filter(lambda x: x.requires_grad, model.parameters())
#optimizer = torch.optim.Adam(params, lr=0.001)
dataloader = get_loader('valid')

best_loss = np.inf
pat = 0
cum_loss = 0
cum_count = 0
it = 0
dataloader.reset_reader()
translations = []
refs = []
while True:
    batch = dataloader.get_data_batch()
    if batch is None:
        break
    if len(batch['text']) == 0:
        continue
    ref = batch['text'][0]
    ref = [dictionary['id2word'][w] for w in ref[1:-1]]
    refs.append(ref)
    tr_list = model.beam_search_topk(batch, 5, 30,k=3)
    translations.append(tr_list)

with open('ref.{}.txt'.format(config.model), 'w') as ref_file:
    for ref in refs:
        ref_file.write('{}\n'.format(' '.join(ref)))

#with open('output.{}.pkl'.format(config.model), 'w') as output_file:
    #for tr in translations:
    #    output_file.write('{}\n'.format(' '.join(tr)))

gen_utils.write_dict_to_pkl(translations,"output.%s.multi.pkl"%config.model)
