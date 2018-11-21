import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader, Dumper
import os
import sys
sys.path.append("../")
import cPickle as pickle
import cv2
import numpy as np
import json
import cv_utils
import shutil

def read_lines_from_text_file(fn):
    with open(fn,"r") as f:
        all_lines=f.readlines()
    all_lines=[x.replace("\n","") for x in all_lines]
    all_lines=[x.replace("\r","") for x in all_lines]
    return all_lines


#   copy all files in src to dst

def copy_all_files_from_src_to_dst(src,dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dst)
    return


#   label list to dict
def label_list2dict(label_list):
    ret_dict={}
    for i in xrange(0,len(label_list)):
        ret_dict[label_list[i]]=i
    return ret_dict

#   save model state dict
def save_model_by_state_dict(model,model_fn):
    import torch
    pass
    torch.save({'state_dict':model.cpu().state_dict()},model_fn)
    model=model.cuda()

#   load model state dict
def load_model_by_state_dict(model,state_dict_fn):
    pass
    from torch.nn import DataParallel
    import torch
    model_dict=torch.load(state_dict_fn)
    try:
        model.load_state_dict(model_dict['state_dict'])
    except:
        try:
            model.load_state_dict(model_dict)
        except:
            model=DataParallel(model.cuda())
            if 'state_dict' in model_dict:
                model.load_state_dict(model_dict['state_dict'])
            else:
                model.load_state_dict(model_dict)
    return model

#   The cv_utils related functions, we still keep the definition here for maximizing the compatibility
def get_frame_set_by_traj(traj):
    return cv_utils.get_frame_set_by_traj(traj)

def bbx_int_area(p_bbx, v_bbx):
    return cv_utils.bbx_int_area(p_bbx,v_bbx)

def bbx_union_area(p_bbx, v_bbx):
    return cv_utils.bbx_union_area(p_bbx,v_bbx)

def calc_spatial_temporal_iou(traj_a,traj_b):
    a,b=cv_utils.calc_spatial_temporal_iou(traj_a,traj_b)
    return a,b

def read_yaml(yaml_filename):
    with open(yaml_filename, "r") as f:
        info = yaml.load(f,Loader=Loader)
    return info

#   write dict or other objects to file
def write_dict_to_pkl(obj,fn):
    if not fn.endswith(".pkl"):
        fn=fn+".pkl"
    with open(fn,"wb") as f:
        pickle.dump(obj,f)

#   read object from dict
def read_dict_from_pkl(fn):
    if not fn.endswith(".pkl"):
        fn=fn+".pkl"

    with open(fn,"rb") as f:
        obj=pickle.load(f)
    return obj

#   write dict or other objects to json
def write_dict_to_json(obj,fn):
    pass
    if not fn.endswith(".json"):
        fn=fn+".json"

    json_str=json.dumps(obj,sort_keys=True,indent=2)
    with open(fn,"wb") as f:
        f.writelines(json_str)
    return

# read selected frames from the video file, only frame within frame_set are selected...
def fast_read_sel_frames_from_vid_file(fn,frame_set,verbose=False,max_fr=None):
    return cv_utils.fast_read_sel_frames_from_vid_file(fn,frame_set,verbose=verbose)

#   read selected frames from the video file, only frame within frame_set are selected...
def read_sel_frames_from_vid_file(fn,frame_set,verbose=False,fast=True,max_fr=None):
    return cv_utils.read_sel_frames_from_vid_file(fn,frame_set,verbose=verbose,fast=fast)

#   read object from json file
def read_dict_from_json(fn):
    if not fn.endswith(".json"):
        fn=fn+".json"

    def byteify(input):
        if isinstance(input, dict):
            return {byteify(key): byteify(value)
                    for key, value in input.iteritems()}
        elif isinstance(input, list):
            return [byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input

    pass
    with open(fn) as json_file:
        obj = json.load(json_file)
    obj=byteify(obj)
    return obj

#   write dict or other objects to both pkl and json format
def write_dict_to_pkl_and_json(obj,fn):
    pass

    fn=fn.replace(".json","")
    fn=fn.replace(".pkl","")

    json_fn=fn+".json"
    pkl_fn=fn+".pkl"

    write_dict_to_json(obj,json_fn)
    write_dict_to_pkl(obj,pkl_fn)

    return

#   list all file with extension
def list_all_files_with_extension(path,extension):
    files=os.listdir(path)
    files_ext=[i for i in files if i.endswith(extension)]
    return files_ext

#   list all folders
def list_all_folders(dir):
    return next(os.walk(dir))[1]

#split a list into n folders
def split_list_into_n_folder_parts(seq,num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def get_frame_by_fnum(vid_full_fn,fnum):
    assert os.path.isfile(vid_full_fn)
    vid_cap = cv2.VideoCapture(vid_full_fn)
    count=0
    prev_fr=None
    while True:
        ret, fr = vid_cap.read()
        if not ret:
            return prev_fr
        if count==fnum:
            return fr
        if count%10==0:
            print vid_full_fn,count
        count += 1
        prev_fr=fr

    return None


#   split a dict into n folders
def split_dict_into_n_folders(raw_dict,n):
    all_list=[]
    for key,val in raw_dict.iteritems():
        all_list.append((key,val))
    sub_lists=split_list_into_n_folder_parts(all_list,n)

    results=[]
    for sl in sub_lists:
        tmp_dict={}
        for meta in sl:
            k=meta[0];v=meta[1]
            tmp_dict[k]=v
        results.append(tmp_dict)
    return results

#split a dict into n folders
def split_dict_into_chunk_size(raw_dict,chunk_size):
    ret_dict_list=[]
    raw_dict_len=len(raw_dict)
    tmp_dict=dict()
    for key,value in raw_dict.iteritems():
        tmp_dict[key]=value
        if len(tmp_dict)==chunk_size:
            ret_dict_list.append(tmp_dict)
            tmp_dict=dict()

    if len(tmp_dict)!=0:
        ret_dict_list.append(tmp_dict)

    return ret_dict_list

split_dict_into_n_dict_list=split_dict_into_chunk_size

#   sort a dictionary by keys
def get_sorted_dict_keys(adict,reverse=False):
    return sorted(adict.keys(),reverse=reverse)

def split_list_dict_into_n_list_dict(raw_dict,chunk_size):
    ret_dict_list=[dict() for _ in xrange(chunk_size)]
    for key,raw_list in raw_dict.iteritems():
        new_list=split_list_into_n_folder_parts(raw_list,chunk_size)
        for i in xrange(chunk_size):
            ret_dict_list[i][key]=new_list[i]
    return ret_dict_list

#   generate probability from svm scores
def from_score_to_prob(score_mat):
    ret_mat=[]
    for now_score in score_mat:
        e_x = np.exp(now_score - np.max(now_score))
        n_ex=e_x / e_x.sum()
        ret_mat.append(n_ex)
    return ret_mat

#   convert object trajectories tuple to object trajectories dictionary
def convert_traj_list_to_traj_dict(traj_list):
    ret_dict=dict()
    for meta in traj_list:
        try:
            ret_dict[meta[0]]=meta[1]
        except:
            continue

    return ret_dict

#   only support mp4 and mpeg
def write_frame_list_to_video_file(out_vid_fn, actv_frames, fr_rate=10, verbose=True):
  return cv_utils.write_frame_list_to_video_file(out_vid_fn, actv_frames, fr_rate, verbose)

#   resize frame list
def resize_frame_list(frame_list,dst_w,dst_h):
    return cv_utils.resize_frame_list(frame_list,dst_w,dst_h)

# image resize
def cv_resize_by_long_edge(im,dst_size=None):
    return cv_utils.cv_resize_by_long_edge(im,dst_size)

def heatmap_caliberation(cam):
    return cv_utils.heatmap_caliberation(cam)

def viz_heatmap_on_img(im,heatmap,hm_weight=0.6):
    pass
    return cv_utils.viz_heatmap_on_img(im,heatmap,hm_weight)

#   read frame list from video file using sklearn
def read_frame_list_from_video_file_sk(vid_fn,verbose=True,max_fr=None):
    return cv_utils.read_frame_list_from_video_file_sk(vid_fn,verbose=verbose,max_fr=max_fr)

def read_frame_list_from_video_file(vid_fn,verbose=True,max_fr=None):
    return cv_utils.read_frame_list_from_video_file(vid_fn,verbose=verbose,max_fr=max_fr)

#   write frame list to images
def write_frame_list_to_imgs(out_vid_fn,actv_frames,verbose=True):
    return cv_utils.write_frame_list_to_imgs(out_vid_fn,actv_frames,verbose)

def calc_spatial_temporal_props_covered_ratio(traj_a, traj_b):
    t,s=cv_utils.calc_spatial_temporal_props_covered_ratio(traj_a,traj_b)
    return t, s

# merge a list of dictionaries
def merge_dict_list(dict_list):
    ret_dict=dict();ret_key_set=set()
    for cur_dict in dict_list:
        for key,val in cur_dict.iteritems():
            if not key in ret_key_set:
                ret_dict[key]=val
                ret_key_set.add(key)
            else:
                raise "Dictionary key overlapped: "+str(key)
    return ret_dict

#   running mean calculation with x and window size N
def runningMeanFast(x, N):
    org_x_len=len(x)
    to_append=x[-N::]
    x=np.concatenate((x,to_append),axis=0)
    conv_x=np.convolve(x, np.ones((N,))/N)[(N-1):]
    ret_x=conv_x[0:org_x_len]
    return ret_x

if __name__=="__main__":
    arr=[10,10.5,10.3,10.2,10.1,10.0,11.5,11.3,10.9]
    arr=np.asarray(arr)
    ret_arr=runningMeanFast(arr,5)