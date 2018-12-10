'''
    Train the noun and verb classifier.
'''

import os
import sys
sys.path.append("../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
from data_loader.GenericDataLoader import DataLoader
from data_loader.NounVerbLoader import NounVerbLoader
import models.BasicNounVerbNN as basic_obj_nn
from torch import nn,optim
import numpy as np
import torch
from torch.autograd import Variable

#token_type="noun"
token_type="verb"

assert token_type in ["noun","verb"]
if token_type=="noun":
    token2id_dict = gen_utils.read_dict_from_pkl("/mnt/hdd_8t/TGIF/noun2id.pkl")
    id2idf_fn="/mnt/hdd_8t/TGIF/noun_id2idf.pkl"

    model_root_path="/mnt/hdd_8t/TGIF/models/noun_det/"
elif token_type=="verb":
    token2id_dict = gen_utils.read_dict_from_pkl("/mnt/hdd_8t/TGIF/verb2id.pkl")
    id2idf_fn="/mnt/hdd_8t/TGIF/verb_id2idf.pkl"

    model_root_path = "/mnt/hdd_8t/TGIF/models/verb_det/"
else:
    raise NotImplementedError

if not os.path.exists(model_root_path):
    os.makedirs(model_root_path)

#   prepare the data loader
dataloader = DataLoader(img_list=gen_utils.read_dict_from_pkl(data_cfg.all_list_fn),batch_size=32,
                        handler_obj=NounVerbLoader(feat_root_path="/mnt/hdd_8t/TGIF/features/jiac_i3d.motion_rgb.obj_det.place365_feat.resnet50",
                                                   text_root_path="/home/wcdu/11731_proj/gt_parsed/",
                                                   token2id_dict=token2id_dict,token_type=token_type))

id2idf=gen_utils.read_dict_from_pkl(id2idf_fn)

#   define the model
model=basic_obj_nn.BasicMultiClassNN(input_layer=7771,num_class=len(token2id_dict))
model=model.cuda()

LEARNING_RATE=0.005
MAX_EPOCH=100
MODEL_FN=os.path.join(model_root_path,"model.pth")

weight_list=[]
for cur_id in sorted(id2idf.keys()):
    weight_list.append(id2idf[cur_id])
weight_list=np.asarray(weight_list)

print "Initialize finished..."

def weighted_binary_cross_entropy_with_logits(logits, targets, pos_weight, weight=None, size_average=True, reduce=True):
    if not (targets.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), logits.size()))

    max_val = (-logits).clamp(min=0)
    log_weight = 1.0 + (pos_weight - 1.0) * targets
    #log_weight_np=log_weight.detach().cpu().numpy()
    #print np.max(log_weight_np),np.min(log_weight_np)
    loss = (1 - targets) * logits + log_weight * (((-max_val).exp() + (-logits - max_val).exp()).log() + max_val)

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, pos_weight, weight=None, size_average=True, reduce=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.weight is not None:
            weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                             pos_weight,
                                                             weight=weight,
                                                             size_average=self.size_average,
                                                             reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                             pos_weight,
                                                             weight=None,
                                                             size_average=self.size_average,
                                                             reduce=self.reduce)

def adjust_learning_rate(init_lr,epoch):

    lr = init_lr * (0.95 ** (epoch // 10))
    lr=max(lr,0.00005)
    return lr

def run_epoch_train(model,dataloader,epoch):
    dataloader.shuffle_data()
    dataloader.reset_reader()

    # set criterion and optimizer
    #criterion=nn.BCEWithLogitsLoss(weight=weight_list)
    criterion=WeightedBCEWithLogitsLoss(pos_weight=torch.from_numpy(weight_list).type(torch.FloatTensor)).cuda()
    optimizer=optim.SGD(model.parameters(),lr=adjust_learning_rate(LEARNING_RATE,epoch),momentum=0.9)

    step=0
    all_loss_list=[]
    while True:
        step+=1
        batch = dataloader.get_data_batch()
        if batch is None:
            break

        cur_feat=batch['feat']
        cur_gt=batch['label']

        model.zero_grad()
        optimizer.zero_grad()

        pred = model.forward(cur_feat)
        loss = criterion(pred, cur_gt)
        loss.backward()
        optimizer.step()

        all_loss_list.append(loss.data.detach().cpu().numpy())
        if step%100==0:
            print "in epoch: ", epoch, "loss: ", loss.data[0],"..."
            #break

    avg_loss=np.nanmean(all_loss_list)
    return model,avg_loss

if __name__=="__main__":
    min_avg_loss=1000000
    for ep in xrange(0,MAX_EPOCH):

        print "In epoch: ",ep,'/',MAX_EPOCH,"..."
        model,cur_avg_loss=run_epoch_train(model,dataloader,ep)

        if cur_avg_loss<min_avg_loss:
            gen_utils.save_model_by_state_dict(model,MODEL_FN)
            min_avg_loss=cur_avg_loss


    print "done."
