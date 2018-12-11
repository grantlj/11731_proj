import torch
import sys
sys.path.append("../")
sys.path.append("../../")
import pickle
import config.dataset_config as data_cfg
import numpy as np
import argparse
from data_loader.GenericDataLoader import DataLoader
from data_loader.SingleVideoPyTorchHandler import VideoTextHandler
from critic_model import *
import utils.gen_utils as gen_utils
from torch.nn.utils import clip_grad_norm


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='baseline.self_critic')
config = parser.parse_args()

dictionary = pickle.load(open('/home/wcdu/11731_proj/dictionary', 'rb'))
txt_root = '/home/wcdu/11731_proj/gt_parsed/'
fast_cider=gen_utils.read_dict_from_pkl("/dev/shm/fast_cider.pkl")
#fast_cider=None
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

if config.model == "baseline.self_critic":
    model = Baseline(300, 1024, 512, 1024, dictionary,fast_cider=fast_cider).cuda()
    du_model = torch.load("/home/wcdu/11731_proj/baseline.pt").cuda()

    model.embed=du_model.embed
    model.encoder=du_model.encoder
    model.decoder=du_model.decoder
    model.word2id=du_model.word2id
    model.id2woord=du_model.id2word

    model.word_dist=du_model.word_dist
    model.drop=du_model.drop


    model.hidden_fc=du_model.hidden_fc
    model.cell_fc=du_model.cell_fc
    model._modules=du_model._modules
    model.drop = nn.Dropout(0)
    model.train()
    print "Model initialization finished..."

else:
    raise NotImplementedError

params=model.parameters()
optimizer = torch.optim.Adam(params, lr=0.00001)
train_dataloader = get_loader('train')
valid_dataloader = get_loader('valid')

best_loss = np.inf
pat = 0
neg_cider=0
for epoch in range(50):

    it = 0
    train_dataloader.shuffle_data()
    train_dataloader.reset_reader()
    model.train()
    cum_loss=0;cum_count=0.0

    ins_sent=None
    while True:

        if it % 200 == 1:
            sys.stderr.write('Train loss: {}\n'.format(cum_loss / cum_count))
            sys.stderr.write('Example sents:{}\n'.format(ins_sent))
            sys.stderr.write('Neg A:{}\n'.format(neg_cider))
        batch = train_dataloader.get_data_batch()
        if batch is None:
            break

        self_critic_loss,num_words,neg_cider,sample_sents,_=model.forward(batch,keep_grad=True)
        ins_sent=sample_sents[0]

        cum_loss += self_critic_loss.item() * num_words
        cum_count += num_words

        optimizer.zero_grad()
        model.zero_grad()
        self_critic_loss.backward()
        clip_grad_norm(params, 0.1)
        optimizer.step()
        it += 1
    


    valid_dataloader.reset_reader()
    model.eval()

    loss_list=[]
    while True:
        batch = valid_dataloader.get_data_batch()
        if batch is None:
            break

        #   use the neg cider for validation
        _, num_words,neg_cider,sample_sents,loss = model(batch, keep_grad=True)
        loss=loss.item()
        loss_list.append(loss)

    avg_loss=np.nanmean(loss_list)
    sys.stderr.write('Dev avg loss: {}\n'.format(avg_loss))

    if avg_loss < best_loss:
        best_loss = avg_loss
        model.faster_cider=None
        torch.save(model, '{}.pt'.format(config.model))
        model.faster_cider=fast_cider
    else:
        pat += 1
    if pat > 3:
        break

