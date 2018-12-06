import os
import sys
sys.path.insert(0,"../")

#import config.dataset_config as data_cfg
import config.coco_dataset_config as data_cfg
import utils.gen_utils as gen_utils
import tensorflow as tf
import numpy as np
import multiprocessing

video_list_fn=data_cfg.all_list_fn
video_list=gen_utils.read_dict_from_pkl(video_list_fn)
model_check_point_filename="/home/jiangl1/data/datasets/TGIF/models/obj_det/faster_rcnn/frozen_inference_graph.pb"
feat_root_path=os.path.join(data_cfg.feat_root_path,"obj_det")
if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

RE_EXTRACT=False
key_frame_interval=1
score_th=0.4

id2label_dict=gen_utils.read_dict_from_pkl("/home/jiangl1/data/datasets/TGIF/models/obj_det/faster_rcnn/id2label.pkl")
print "Initialize finished..."

def sample_list_with_interval(vid_frames,key_frame_interval):
    ret_list=[]
    for i in xrange(0,len(vid_frames),key_frame_interval):
        ret_list.append(vid_frames[i])
    vid_frames=None
    return ret_list

def handle_a_particular_list(vid_list,gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    # load tensorflow model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_check_point_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print "Model restored:", model_check_point_filename

    with tf.Session(graph=detection_graph, config=gpu_config) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        category_tensor= detection_graph.get_tensor_by_name('detection_classes:0')
        for vid in vid_list:
            vid_fn=os.path.join(data_cfg.video_root_path,str(vid)+".mp4")
            vid_feat_fn=os.path.join(feat_root_path,str(vid)+".pkl")
            if os.path.isfile(vid_feat_fn) and not RE_EXTRACT:
                continue

            if not os.path.isfile(vid_fn):
                continue

            vid_frames=gen_utils.read_frame_list_from_video_file_sk(vid_fn,verbose=True,max_fr=5000)
            vid_frames=sample_list_with_interval(vid_frames,key_frame_interval=key_frame_interval)

            vid_det_dict={}
            fr_ind=0
            for fr in vid_frames:

                h=len(fr);w=len(fr[0])
                det_list=[]
                print "video: ",vid," fr: ",fr_ind," gpuid: ",gpu_id
                im_expanded = np.expand_dims(fr, axis=0)


                (boxes, scores,categories) = sess.run(
                    [boxes_tensor, scores_tensor,category_tensor],
                    feed_dict={image_tensor: im_expanded})
                boxes = np.squeeze(boxes);
                scores = np.squeeze(scores)
                categories=np.squeeze(categories)

                # selected bounding boxes which scores are larger than threshold
                sel_boxes = boxes[np.argwhere(scores > score_th)]
                sel_scores = scores[np.argwhere(scores > score_th)]
                sel_categories=categories[np.argwhere(scores > score_th)].tolist()

                for i in xrange(0,len(sel_categories)):
                    box=sel_boxes[i,:][0]
                    score=sel_scores[i,:][0]
                    cat=int(sel_categories[i][0])

                    x1=int(box[0]*w);y1=int(box[1]*h);x2=int(box[2]*w);y2=int(box[3]*h)
                    box=[x1,y1,x2,y2]

                    det_list.append({'box':box,'score':score,'cat':id2label_dict[cat]})
                    #print "done"

                vid_det_dict[fr_ind]=det_list
                fr_ind += key_frame_interval

            gen_utils.write_dict_to_pkl(vid_det_dict,vid_feat_fn)



    return

if __name__=="__main__":
    gpu_id_list=[6,7];num_gpu=len(gpu_id_list)
    sub_vid_lists=gen_utils.split_list_into_n_folder_parts(video_list,num_gpu)

    thread_pool=[]
    for vid_list,gpu_id in zip(sub_vid_lists,gpu_id_list):
        th=multiprocessing.Process(target=handle_a_particular_list,args=(vid_list,gpu_id,))
        th.start()
        thread_pool.append(th)


    for th in thread_pool:
        th.join()

    print "done."