import os

#   output sed format score files...
def generate_csv_file(csv_root_path,vid_name,vid_result_list):
    pass
    all_list = ['"ID","EventType","Framespan","DetectionScore","DetectionDecision"\n']
    ind=0
    for re_list in vid_result_list:
        ind+=1
        now_line = '"%d","%s","%d:%d","%f","%d"\n' % (ind, re_list[0], re_list[1], re_list[2], re_list[3], re_list[4])
        all_list.append(now_line)

    csv_filename=os.path.join(csv_root_path,vid_name+".csv")
    with open(csv_filename,"w") as f:
        f.writelines(all_list)

