import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils


gt_root_path=data_cfg.gt_root_path
lctx_root_path=data_cfg.long_ctx_root_path
sctx_root_path=data_cfg.short_ctx_root_path
q_root_path=data_cfg.q_root_path


def load_quaser_gt_by_id(id):
    pass
    gt_fn=os.path.join(gt_root_path,str(id)+".pkl")
    gt=gen_utils.read_dict_from_pkl(gt_fn)
    return gt

def load_quaser_lctx_by_id(id):
    pass
    lctx_fn = os.path.join(lctx_root_path, str(id) + ".pkl")
    lctx = gen_utils.read_dict_from_pkl(lctx_fn)
    return lctx

def load_quaser_sctx_by_id(id):
    pass
    sctx_fn = os.path.join(sctx_root_path, str(id) + ".pkl")
    sctx = gen_utils.read_dict_from_pkl(sctx_fn)
    return sctx

def load_quaser_qmeta_by_id(id):
    pass
    q_fn = os.path.join(q_root_path, str(id) + ".pkl")
    q = gen_utils.read_dict_from_pkl(q_fn)
    return q

#   the data utility functions for quaser dataset
def load_quaser_all_by_id(id):
    gt_text=load_quaser_gt_by_id(id)
    lctx_text=load_quaser_lctx_by_id(id)
    sctx_text=load_quaser_sctx_by_id(id)
    q_meta=load_quaser_qmeta_by_id(id)

    return gt_text,lctx_text,sctx_text,q_meta


def read_corpus(file_path):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        data.append(sent)

    return data
