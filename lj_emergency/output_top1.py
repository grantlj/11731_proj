import os
import sys
sys.path.append("../")
import utils.gen_utils as gen_utils

org_fn="output.baseline_template_lj.multi.pkl"
dst_fn="pred.baseline_template_lj.txt"
if __name__=="__main__":
    dst_lines=[]
    org_list=gen_utils.read_dict_from_pkl(org_fn)
    for meta in org_list:
        dst_lines.append(" ".join(meta[0]))
    dst_lines=[x+"\n" for x in dst_lines]
    with open(dst_fn,"w") as f:
        f.writelines(dst_lines)

    print "done."