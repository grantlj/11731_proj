import torch
import os
import sys
sys.path.append("../")
sys.path.append("../../")
import pickle
import config.dataset_config as data_cfg
import numpy as np
import argparse
from data_loader.GenericDataLoader import DataLoader
from data_loader.SingleVideoPyTorchHandler import VideoTextHandler


parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str, default='baseline.self_critic')
config = parser.parse_args()
txt_root = '/home/wcdu/11731_proj/gt_parsed/'
dictionary = pickle.load(open('/home/wcdu/11731_proj/dictionary', 'rb'))
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

model = torch.load('{}.pt'.format(config.model)).cuda()

#dataloader = get_loader('valid')
dataloader= get_loader('valid')

best_loss = np.inf
pat = 0
cum_loss = 0
cum_count = 0
it = 0
dataloader.reset_reader()
translations = []
refs = []

step=0
while True:
    print "In prediction, step: ",step
    step+=1
    #if step>10:
    #    break
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

print "Dumping ref..."
with open('ref.{}.txt'.format(config.model), 'w') as ref_file:
    for ref in refs:
        ref_file.write('{}\n'.format(' '.join(ref)))

print "Dumping predictions..."
with open('output.{}.txt'.format(config.model), 'w') as output_file:
    for tr in translations:
        output_file.write('{}\n'.format(' '.join(tr)))
print "done."
