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
        raise NotImplementedError
    dataloader = DataLoader(img_list=pickle.load(open(cfg_list_fn, 'rb')),
                            batch_size=16,
                            num_worker=4,
                            handler_obj=VideoTextHandler(video_root_path="../TGIF/features/resnet50", key_frame_interval=8, text_root_path='../TGIF/gt', dic=dictionary['word2id']))
    return dataloader

model = Baseline(300, 2048, 512, 512, dictionary).cuda()
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
    while True:
        if it % 500 == 1:
            sys.stderr.write('Train loss: {}\n'.format(cum_loss / cum_count))
        batch = train_dataloader.get_data_batch()
        if batch is None:
            break
        loss, num_words = model(batch, True)
        cum_loss += loss.item() * num_words
        cum_count += num_words
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(params, 0.1)
        optimizer.step()
        it += 1
    val_loss = 0
    val_count = 0
    valid_dataloader.reset_reader()
    while True:
        batch = valid_dataloader.get_data_batch()
        if batch is None:
            break
        loss, num_words = model(batch, True)
        val_loss += loss.item() * num_words
        val_count += num_words
    sys.stderr.write('Dev loss: {}\n'.format(val_loss / val_count))
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, 'baseline.pt')
    else:
        pat += 1
    if pat > 3:
        break

