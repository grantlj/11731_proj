import config.dataset_config as data_cfg
import sys
from data_loader.GenericDataLoader import DataLoader
from data_loader.SingleVideoPyTorchHandler import VideoTextHandler
from model import *
from torch.nn.utils import clip_grad_norm
import pickle

dictionary = pickle.load(open('dictionary', 'rb'))
txt_root = './gt_parsed'


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
                            handler_obj=VideoTextHandler(video_root_path="/mnt/hdd_8t/TGIF/features/resnet50", key_frame_interval=8, text_root_path=txt_root, dic=dictionary['word2id']))
    return dataloader

def convert_text2textstr(text):
    text=text[1:len(text)-1]
    ret=[dictionary['id2word'][x] for x in text]
    return ret

if __name__=="__main__":
    data_loader=get_loader("train")
    data_loader.shuffle_data()
    data_loader.reset_reader()

    ind=0
    while True:
        print ind
        batch=data_loader.get_data_batch()
        if batch is None:
            break

        id_list=batch['id_list']
        text_list=batch['text']

        for ind,text in zip(id_list,text_list):
            text_str=convert_text2textstr(text)
            print ind,text_str

        break

    print "done."

