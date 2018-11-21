'''
    The image and video related utilities...
'''

import cv2
import numpy as np
import os

def get_frame_set_by_traj(traj):
    ret_set=set(traj.keys())
    return ret_set

def bbx_int_area(p_bbx, v_bbx):
    if p_bbx==[] or v_bbx==[]:
        return 0
    try:
        x = max(p_bbx[0], v_bbx[0])
    except:
        return 0
    y = max(p_bbx[1], v_bbx[1])

    w = min(p_bbx[2], v_bbx[2]) - x
    h = min(p_bbx[3], v_bbx[3]) - y

    if w <= 0 or h <= 0:
        return 0
    else:
        return w*h

def bbx_union_area(p_bbx, v_bbx):
    if p_bbx==[] or v_bbx==[]:
        return 0

    x = min(p_bbx[0], v_bbx[0])
    y = min(p_bbx[1], v_bbx[1])
    w = max(p_bbx[2], v_bbx[2]) - x
    h = max(p_bbx[3], v_bbx[3]) - y

    return w*h


def calc_spatial_temporal_iou(traj_a,traj_b):

    def get_sub_traj_by_frame_set(org_traj,fr_set):
        ret_traj=[]
        for fr,bbx_meta in org_traj.iteritems():
            if bbx_meta==[]:
                bbx_meta=[1,1,2,2]
            if fr in fr_set:
                ret_traj.append((fr,bbx_meta))
                fr_set.remove(fr)
        ret_traj=sorted(ret_traj)
        return ret_traj


    pass
    traj_a_fr_set=get_frame_set_by_traj(traj_a)
    traj_b_fr_set=get_frame_set_by_traj(traj_b)

    ab_int_set=traj_a_fr_set.intersection(traj_b_fr_set)
    ab_union_set=traj_a_fr_set.union(traj_b_fr_set)

    t_int=len(ab_int_set);t_uni=len(ab_union_set)
    t_iou=(t_int+0.0)/(t_uni+0.0)

    if t_iou==0:
        return t_iou,0

    traj_a=get_sub_traj_by_frame_set(traj_a,ab_int_set)
    ab_int_set=traj_a_fr_set.intersection(traj_b_fr_set)
    traj_b=get_sub_traj_by_frame_set(traj_b,ab_int_set)

    a_bbx_list=[]
    b_bbx_list=[]

    for i in xrange(len(traj_a)):
        a_bbx_list.append(traj_a[i][1])
        b_bbx_list.append(traj_b[i][1])

    a_bbx_list=np.asarray(a_bbx_list)
    b_bbx_list=np.asarray(b_bbx_list)

    if len(traj_a)==0:
        return t_iou,0

    try:
        a_avg_bbx=np.mean(a_bbx_list,axis=0)
        b_avg_bbx=np.mean(b_bbx_list,axis=0)
    except:
        print 'error'

    try:
        s_iou=bbx_int_area(a_avg_bbx,b_avg_bbx)/float(bbx_union_area(a_avg_bbx,b_avg_bbx))
    except:
        s_iou=0

    return t_iou,s_iou

# read selected frames from the video file, only frame within frame_set are selected...
def fast_read_sel_frames_from_vid_file(fn,frame_set,verbose=False):

    all_frames_dict=dict()
    vid_cap=cv2.VideoCapture(fn)
    cnt=0
    while True:
        #ret,fr=vid_cap.read()
        ret=vid_cap.grab()
        cnt+=1
        if cnt%500==0 and verbose:
            print fn,":",cnt
        if ret==False:
            break

        if cnt in frame_set:
            ret,fr=vid_cap.retrieve()

            if ret==False:
                break

            all_frames_dict[cnt]=fr
            frame_set.remove(cnt)

        if 0 in frame_set and cnt==1:
            ret,fr = vid_cap.retrieve()

            if ret==False:
                break

            all_frames_dict[0]=fr
            frame_set.remove(0)
        if len(frame_set)==0:
            break

    return all_frames_dict

#   read selected frames from the video file, only frame within frame_set are selected...
def read_sel_frames_from_vid_file(fn,frame_set,verbose=False,fast=True):
    if fast:
        return fast_read_sel_frames_from_vid_file(fn,frame_set,verbose=verbose)

    all_frames_dict=dict()
    vid_cap=cv2.VideoCapture(fn)
    cnt=0
    while True:
        ret,fr=vid_cap.read()
        cnt+=1
        if cnt%500==0 and verbose:
            print fn,":",cnt
        if ret==False:
            break
        if cnt in frame_set:
            all_frames_dict[cnt]=fr
        if 0 in frame_set and cnt==1:
            all_frames_dict[0]=fr

    return all_frames_dict

#   only support mp4 and mpeg
def write_frame_list_to_video_file(out_vid_fn, actv_frames, fr_rate=10, verbose=True):
  if out_vid_fn.endswith('mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'X264')
  elif out_vid_fn.endswith('avi'):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  else:
      raise NotImplementedError

  width = int(len(actv_frames[0][0]));height = int(len(actv_frames[0]))
  out_vid_cap = cv2.VideoWriter(out_vid_fn, fourcc, fr_rate, (width, height))

  count=0
  for fr in actv_frames:
    if verbose and count%50==0:
      print out_vid_fn," done: ",count, "frames..."
    count += 1
    out_vid_cap.write(fr)
  out_vid_cap.release()

#   calc how much trajectory_a is covered by trajectory_b
def calc_spatial_temporal_props_covered_ratio(traj_a, traj_b):

    def get_sub_traj_by_frame_set(org_traj, fr_set):
        ret_traj = []
        for bbx_meta in org_traj:
            fr = bbx_meta[0]
            if fr in fr_set:
                ret_traj.append(bbx_meta)
                fr_set.remove(fr)
        ret_traj = sorted(ret_traj)
        return ret_traj

    pass
    traj_a_fr_set = get_frame_set_by_traj(traj_a)
    traj_b_fr_set = get_frame_set_by_traj(traj_b)

    ab_int_set = traj_a_fr_set.intersection(traj_b_fr_set)

    t_int = len(ab_int_set)
    t_covered_ratio=t_int/float(len(traj_a_fr_set))

    if t_covered_ratio == 0:
        return t_covered_ratio, 0

    traj_a = get_sub_traj_by_frame_set(traj_a, ab_int_set)
    ab_int_set = traj_a_fr_set.intersection(traj_b_fr_set)
    traj_b = get_sub_traj_by_frame_set(traj_b, ab_int_set)

    a_bbx_list = []
    b_bbx_list = []

    for i in xrange(len(traj_a)):
        if traj_a[i][1]!=[]:
            a_bbx_list.append(traj_a[i][1])
        if traj_b[i][1]!=[]:
            b_bbx_list.append(traj_b[i][1])

    a_bbx_list = np.asarray(a_bbx_list)
    b_bbx_list = np.asarray(b_bbx_list)

    a_avg_bbx = np.mean(a_bbx_list, axis=0)
    try:
        b_avg_bbx = np.mean(b_bbx_list, axis=0)
    except:
        return t_covewred_ratio,t_covered_ratio
    s_covered_ratio = bbx_int_area(a_avg_bbx, b_avg_bbx) / float(bbx_union_area(a_avg_bbx, a_avg_bbx))

    return t_covered_ratio, s_covered_ratio

#   resize frame list
def resize_frame_list(frame_list,dst_w,dst_h):
    ret_fr_list=[]
    for fr in frame_list:
        fr=cv2.resize(fr,(dst_w,dst_h),cv2.INTER_LINEAR)
        ret_fr_list.append(fr)

    return ret_fr_list

# image resize
def cv_resize_by_long_edge(im,dst_size=None):
    org_h, org_w, _ = im.shape

    if dst_size != None:
        if org_h > org_w:
            im = cv2.resize(im, (int(dst_size / float(org_h) * org_w), int(dst_size)))
        else:
            im = cv2.resize(im, (dst_size, int(dst_size / float(org_w) * org_h)))

    return im

def write_frame_list_to_imgs(out_vid_fn,actv_frames,verbose=True):
    out_vid_path=out_vid_fn.replace(".mp4","")
    if not os.path.exists(out_vid_path):
        os.mkdir(out_vid_path)

    count=0
    for fr in actv_frames:
        img_fn=os.path.join(out_vid_path,"%06d.jpg"%count)
        cv2.imwrite(img_fn,fr)
        count += 1
    return


#   read frame list from video file
def read_frame_list_from_video_file_sk(vid_fn,verbose=True,max_fr=None):
    #from skvideo.io import VideoCapture
    import skvideo.io
    vid_cap=skvideo.io.vreader(vid_fn)
    count=0
    ret_fr=[]
    for fr in vid_cap:
        count+=1
        if verbose and count%50==0:
            print vid_fn, " done: ",count, "frames..."
        ret_fr.append(fr)
        if not max_fr is None and count>max_fr:
            break

    try:
        vid_cap.close()
    except:
        pass

    try:
        vid_cap.release()
    except:
        pass

    try:
        fr=None
    except:
        pass
    return ret_fr

#   read frame list from video file
def read_frame_list_from_video_file(vid_fn,verbose=True,max_fr=None):
    vid_cap=cv2.VideoCapture(vid_fn)
    count=0
    ret_fr=[]
    while True:
        ret,fr=vid_cap.read()
        if ret==False:
            break
        count+=1
        if verbose and count%50==0:
            print vid_fn, " done: ",count, "frames..."
        ret_fr.append(fr)
        if not max_fr is None and count>max_fr:
            break

    try:
        vid_cap.close()
    except:
        pass

    try:
        vid_cap.release()
    except:
        pass

    try:
        fr=None
    except:
        pass
    return ret_fr

def heatmap_caliberation(cam):
    cam=cam-np.min(cam)
    cam=cam/np.max(cam)
    cam=np.uint8(255*cam)
    return cam

def viz_heatmap_on_img(im,heatmap,hm_weight=0.6):
    pass
    heatmap = heatmap_caliberation(heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = hm_weight * colormap + (1.0-hm_weight) * im
    return result