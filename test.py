import torch
import pdb
import pickle
import config.dataset_config as data_cfg
import numpy as np
import sys
from data_loader.GenericDataLoader import DataLoader
from data_loader.SingleVideoPyTorchHandler import VideoTextHandler
from model import Baseline
from torch.nn.utils import clip_grad_norm


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
                            handler_obj=VideoTextHandler(video_root_path="../TGIF/features/resnet50", key_frame_interval=8, text_root_path='../TGIF/gt', dic=dictionary['word2id']))
    return dataloader

#model = Baseline(300, 2048, 512, 512, dictionary).cuda()
model = torch.load('baseline.pt').cuda()
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
    ref = batch['text'][0]
    ref = [dictionary['id2word'][w] for w in ref[1:-1]]
    refs.append(ref)
    tr = model.beam_search(batch, 10, 20)
    translations.append(tr)

with open('ref.txt', 'w') as ref_file:
    for ref in refs:
        ref_file.write('{}\n'.format(' '.join(ref)))

with open('output.txt', 'w') as output_file:
    for tr in translations:
        output_file.write('{}\n'.format(' '.join(tr)))
