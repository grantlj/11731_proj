import os
import sys
sys.path.append("../")
import config.dataset_config as data_cfg
import utils.gen_utils as gen_utils
import pickle
dictionary = pickle.load(open('/home/wcdu/11731_proj/dictionary', 'rb'))

noun2id_fn="/mnt/hdd_8t/TGIF/noun2id.pkl"
id2noun_fn="/mnt/hdd_8t/TGIF/id2noun.pkl"
noun_idf_fn="/mnt/hdd_8t/TGIF/noun_id2idf.pkl"

verb2id_fn="/mnt/hdd_8t/TGIF/verb2id.pkl"
id2verb_fn="/mnt/hdd_8t/TGIF/id2verb.pkl"
verb_idf_fn="/mnt/hdd_8t/TGIF/verb_id2idf.pkl"

def iterate_over_meta(all_meta):
    obj2id={}
    id2obj={}
    id2idf={}

    cur_id=0;total_df=0;cnt_name_list=[]
    for obj,val in all_meta.iteritems():
        obj2id[obj]=cur_id
        id2obj[cur_id]=obj
        id2idf[cur_id]=val
        cnt_name_list.append((val,obj))
        total_df+=val
        cur_id+=1
    cnt_name_list=sorted(cnt_name_list,reverse=True)
    print cnt_name_list[0:3]

    for cur_id in id2idf.keys():
        id2idf[cur_id]=1/(float(id2idf[cur_id])/float(total_df))

    return obj2id,id2obj,id2idf

if __name__=="__main__":
    noun_meta=dictionary['nouns']
    verb_meta=dictionary['verbs']

    noun2id,id2noun,noun_idf=iterate_over_meta(noun_meta)
    verb2id,id2verb,verb_idf=iterate_over_meta(verb_meta)

    gen_utils.write_dict_to_pkl(noun2id,noun2id_fn)
    gen_utils.write_dict_to_pkl(id2noun,id2noun_fn)
    gen_utils.write_dict_to_pkl(noun_idf,noun_idf_fn)

    gen_utils.write_dict_to_pkl(verb2id,verb2id_fn)
    gen_utils.write_dict_to_pkl(id2verb,id2verb_fn)
    gen_utils.write_dict_to_pkl(verb_idf,verb_idf_fn)
    print "done."