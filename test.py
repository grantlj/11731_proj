import torch
import pdb
import pickle
import config.dataset_config as data_cfg
import numpy as np
import sys
import argparse
from data_loader.GenericDataLoader import DataLoader
from data_loader.SingleVideoPyTorchHandler import VideoTextHandler
from model import Baseline
from torch.nn.utils import clip_grad_norm


parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str, default='baseline') 
config = parser.parse_args()

dictionary = pickle.load(open('dictionary', 'rb'))
def get_loader(split):
    if split == 'train':
        cfg_list_fn = data_cfg.trn_list_fn
    elif split == 'valid':
        cfg_list_fn = data_cfg.val_list_fn
    else:
        cfg_list_fn = data_cfg.tst_list_fn
    dataloader = DataLoader(img_list=pickle.load(open(cfg_list_fn, 'rb')),
                            batch_size=1,
                            num_worker=4,
                            handler_obj=VideoTextHandler(video_root_path="../TGIF/features/jiac_i3d", 
                                                         key_frame_interval=8, 
                                                         text_root_path='./gt_parsed', 
                                                         obj_det_path='../TGIF/features/obj_det',
                                                         mot_rec_path='../TGIF/features/motion_rgb',
                                                         place_path='../TGIF/features/place365_feat',
                                                         frame_path='../TGIF/features/resnet50',
                                                         dic=dictionary['word2id']))
    return dataloader

model = torch.load('{}.pt'.format(config.model)).cuda()
#params = filter(lambda x: x.requires_grad, model.parameters())
#optimizer = torch.optim.Adam(params, lr=0.001)
dataloader = get_loader('test')

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
    tr = model.beam_search(batch, 5, 30)
    translations.append(tr)

with open('ref.{}.txt'.format(config.model), 'w') as ref_file:
    for ref in refs:
        ref_file.write('{}\n'.format(' '.join(ref)))

with open('output.{}.txt'.format(config.model), 'w') as output_file:
    for tr in translations:
        output_file.write('{}\n'.format(' '.join(tr)))
