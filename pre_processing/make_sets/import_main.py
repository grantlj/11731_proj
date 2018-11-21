'''
   Make the train/validation/test sets.
'''

import os
import sys
sys.path.append('../')
sys.path.append("../../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
import urllib as req
import threading
import moviepy.editor as mp



ins_list_fn="/home/jiangl1/data/datasets/TGIF/TGIF-Release/data/tgif-v1.0.tsv"
ins_list=gen_utils.read_lines_from_text_file(ins_list_fn)

org_gif_root_path="/home/jiangl1/data/datasets/TGIF/raw/gif/"
dst_video_root_path=data_cfg.video_root_path

org_split_root_path="/home/jiangl1/data/datasets/TGIF/TGIF-Release/data/splits/"
org_trn_fn=os.path.join(org_split_root_path,"train.txt")
org_val_fn=os.path.join(org_split_root_path,"val.txt")
org_tst_fn=os.path.join(org_split_root_path,"test.txt")

org_trn_lst=set(gen_utils.read_lines_from_text_file(org_trn_fn))
org_val_lst=set(gen_utils.read_lines_from_text_file(org_val_fn))
org_tst_lst=set(gen_utils.read_lines_from_text_file(org_tst_fn))

trn_list=gen_utils.read_dict_from_pkl(data_cfg.trn_list_fn)
val_list=gen_utils.read_dict_from_pkl(data_cfg.val_list_fn)
tst_list=gen_utils.read_dict_from_pkl(data_cfg.tst_list_fn)

all_list=trn_list+val_list+tst_list

gen_utils.write_dict_to_pkl(all_list,data_cfg.all_list_fn)

print "Initialize finished..."

MAX_TH=18

def convert_gif_to_mp4(id):
    org_gif_fn=os.path.join(org_gif_root_path,str(id)+".gif")
    print org_gif_fn
    assert os.path.isfile(org_gif_fn)


    dst_mp4_fn=os.path.join(dst_video_root_path,str(id)+".mp4")

    if os.path.isfile(dst_mp4_fn):
        return


    cmd = "ffmpeg -f gif -i %s %s" % (org_gif_fn, dst_mp4_fn)
    print cmd

    os.system(cmd)
    return

def filter_by_mp4(org_list):
    ret_list=[]
    for id in org_list:
        mp4_fn = os.path.join(dst_video_root_path, str(id) + ".mp4")
        if os.path.isfile(mp4_fn):
            ret_list.append(id)
    return ret_list

#   filter out irreleveant items
if __name__=="__main__":


    new_trn_ids = filter_by_mp4(trn_list)
    new_val_ids = filter_by_mp4(val_list)
    new_tst_ids = filter_by_mp4(tst_list)

    new_all_ids=new_trn_ids+new_val_ids+new_tst_ids
    gen_utils.write_dict_to_pkl(new_trn_ids,data_cfg.trn_list_fn)
    gen_utils.write_dict_to_pkl(new_val_ids,data_cfg.val_list_fn)
    gen_utils.write_dict_to_pkl(new_tst_ids,data_cfg.tst_list_fn)

    print "done."

'''
if __name__=="__main__":

    thread_pool=[]

    convert_gif_to_mp4(4)

    for id in all_list:


        th=threading.Thread(target=convert_gif_to_mp4,args=(id,))
        th.start()
        thread_pool.append(th)
        while len(threading.enumerate())>=MAX_TH:
            pass
        #convert_gif_to_mp4(id)

    for th in thread_pool:
        th.join()
    print "done."
'''

'''
if __name__=="__main__":

    trn_list=[]
    val_list=[]
    tst_list=[]

    for i in xrange(0,len(ins_list)):
        org_gif_fn=os.path.join(org_gif_root_path,str(i)+".gif")
        if not os.path.isfile(org_gif_fn):
            print i,org_gif_fn," NOT FOUND..."

        ins_meta=ins_list[i]
        tmp_str=ins_meta.split("\t")
        url=tmp_str[0];gt=tmp_str[1]

        if url in org_trn_lst:
            trn_list.append(i)
        if url in org_val_lst:
            val_list.append(i)
        if url in org_tst_lst:
            tst_list.append(i)

    gen_utils.write_dict_to_pkl(trn_list,data_cfg.trn_list_fn)
    gen_utils.write_dict_to_pkl(val_list,data_cfg.val_list_fn)
    gen_utils.write_dict_to_pkl(tst_list,data_cfg.tsta_list_fn)

    print "done."

'''


