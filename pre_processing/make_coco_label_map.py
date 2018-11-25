import sys
sys.path.append("../")
import utils.gen_utils as gen_utils

org_fn="/home/jiangl1/data/datasets/TGIF/models/obj_det/faster_rcnn/lj_label.txt"

with open(org_fn) as f:
    all_lines=f.readlines()
all_lines=[x.replace("\n","") for x in all_lines]
all_lines=[x.replace("    ",",") for x in all_lines]

id2name_dict={}
for line in all_lines:
    tmp_str=line.split(",")
    id=int(tmp_str[0])
    name=tmp_str[1]
    id2name_dict[id]=name

gen_utils.write_dict_to_pkl(id2name_dict,"/home/jiangl1/data/datasets/TGIF/models/obj_det/faster_rcnn/id2label.pkl")
print "done."
