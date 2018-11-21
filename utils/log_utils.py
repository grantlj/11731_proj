'''
    The logger utility for logging the intermediate results
'''

import os
import sys
sys.path.append("../")
import utils.gen_utils as gen_utils

class LJLogger(object):
    def __init__(self,dump_root_path,auto_dump=False,load_history=False):
        pass
        if not os.path.exists(dump_root_path):
            os.makedirs(dump_root_path)

        self.dump_file=os.path.join(dump_root_path,"log.log")
        if load_history and os.path.isfile(self.dump_file):
            self.dump_dict=gen_utils.read_dict_from_json(self.dump_file)
        else:
            self.dump_dict=dict()
        self.auto_dump=auto_dump

    def log(self,time,item_val_dict):
        if not time in self.dump_dict:
            self.dump_dict[time]=dict()

        for item,val in item_val_dict.iteritems():
            self.dump_dict[time][item]=val

        if self.auto_dump:
            self.dump()

    def dump(self):
        gen_utils.write_dict_to_json(self.dump_dict,self.dump_file)
        return
