'''
    Download the TGIF dataset.
'''

import os
import sys
sys.path.append('../')
sys.path.append("../../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
import urllib as req
import threading

MAX_TH=4

ins_list_fn="/home/jiangl1/data/datasets/TGIF/TGIF-Release/data/tgif-v1.0.tsv"
ins_list=gen_utils.read_lines_from_text_file(ins_list_fn)

dst_gif_root_path=os.path.join(data_cfg.raw_data_root_path,"gif")
dst_gt_root_path=os.path.join(data_cfg.raw_data_root_path,"gt")

if not os.path.exists(dst_gif_root_path):
    os.makedirs(dst_gif_root_path)
if not os.path.exists(dst_gt_root_path):
    os.makedirs(dst_gt_root_path)

def download_a_line(line_id,line_meta):
    print line_id
    tmp_str=line_meta.split("\t")
    url=tmp_str[0]
    gt=tmp_str[1]

    dst_gif_fn=os.path.join(dst_gif_root_path,str(line_id)+".gif")
    dst_gt_fn=os.path.join(dst_gt_root_path,str(line_id)+".pkl")

    try:
        req.urlretrieve(url, dst_gif_fn)
    except:
        if os.path.isfile(dst_gif_fn):
            os.remove(dst_gif_fn)

    gen_utils.write_dict_to_pkl(gt,dst_gt_fn)

    return

if __name__=="__main__":
    thread_pool=[]

    for i in xrange(0,len(ins_list)):
        line_id=i
        line_meta=ins_list[line_id]

        th=threading.Thread(target=download_a_line,args=(line_id,line_meta,))
        th.start()
        thread_pool.append(th)

        while len(threading.enumerate())>=MAX_TH:
            pass

        #download_a_line(line_id,line_meta)

    for th in thread_pool:
        th.join()

    print "done."