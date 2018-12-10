import torch
import pdb
import pickle
import config.dataset_config as data_cfg
import numpy as np
import sys
import argparse
from data_loader.GenericDataLoader import DataLoader
from data_loader.SingleVideoPyTorchHandler import VideoTextHandler
from model import *
from torch.nn.utils import clip_grad_norm


parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str, default='baseline_template_lj')
config = parser.parse_args() 

dictionary = pickle.load(open('/home/wcdu/11731_proj/template_pos_dict.pkl', 'rb'))
#txt_root = '../TGIF/gt'
txt_root = '/home/wcdu/11731_proj/gt_parsed_nv_template/'
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

if config.model == 'baseline' or config.model=='baseline_template_lj':
    model = Baseline(300, 1024, 512, 1024, dictionary).cuda()
elif config.model == 'mos':
	model = MoS(300, 1024, 512, 1024, dictionary).cuda()
elif config.model == 'mos_ext':
	model = MoS_EXT(300, 1024, 512, 1024, dictionary).cuda()

for p in model.parameters():
    torch.nn.init.uniform_(p.data, a=-0.1, b=0.1)
params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)
train_dataloader = get_loader('train')
valid_dataloader = get_loader('valid')

best_loss = np.inf
pat = 0
for epoch in range(50):
    cum_loss = 0
    cum_count = 0
    it = 0
    #params = filter(lambda x: x.requires_grad, model.parameters())
    #optimizer = torch.optim.SGD(params, lr=0.1 * 0.9 ** epoch)
    train_dataloader.shuffle_data()
    train_dataloader.reset_reader()
    model.train()
    while True:
        if it % 500 == 1:
            sys.stderr.write('Train loss: {}\n'.format(cum_loss / cum_count))
        batch = train_dataloader.get_data_batch()
        if batch is None:
            break
        if  'baseline' in config.model:
            loss, num_words = model(batch, True)
        else:
            loss, num_words, nn_loss, num_nn = model(batch, True)
        cum_loss += loss.item() * num_words
        cum_count += num_words
        optimizer.zero_grad()
        if config.model in ['mos', 'mos_ext']:
            #loss += nn_loss
            pass
        loss.backward()
        clip_grad_norm(params, 0.1)
        optimizer.step()
        it += 1

    val_loss = 0
    val_count = 0
    valid_dataloader.reset_reader()
    model.eval()
    while True:
        batch = valid_dataloader.get_data_batch()
        if batch is None:
            break
        if 'baseline' in config.model:
            loss, num_words = model(batch, True)
        else:
            loss, num_words, nn_loss, num_nn = model(batch, True)
        val_loss += loss.item() * num_words
        val_count += num_words
    sys.stderr.write('Dev loss: {}\n'.format(val_loss / val_count))
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, '{}.pt'.format(config.model))
    else:
        pat += 1
    if pat > 3:
        break

