'''
    The virat 2.0 dataset utility functions.
'''

import os
import sys
sys.path.append("../")

obj_id_type_dict={0:"Other",1:"Person",2:"Vehicle",3:"Vehicle",4:"Other",5:"Bike"}

'''
Event Type ID (for column 2 above)
1: Person loading an Object to a Vehicle
2: Person Unloading an Object from a Car/Vehicle
3: Person Opening a Vehicle/Car Trunk
4: Person Closing a Vehicle/Car Trunk
5: Person getting into a Vehicle
6: Person getting out of a Vehicle
7: Person gesturing
8: Person digging
9: Person carrying an object
10: Person running
11: Person entering a facility
12: Person exiting a facility

'''
event_id_type_dict={
    1:"Loading",
    2:"Unloading",
    3:"Open_Trunk",
    4:"Closing_Trunk",
    5:"Entering",
    6:"Exiting",
    7:None,
    8:None,
    9:None,
    10:None,
    11:"Entering",
    12:"Exiting",
}

org_event_id_type_dict={
1: "Person loading an Object to a Vehicle",
2: "Person Unloading an Object from a Car/Vehicle",
3: "Person Opening a Vehicle/Car Trunk",
4: "Person Closing a Vehicle/Car Trunk",
5: "Person getting into a Vehicle",
6: "Person getting out of a Vehicle",
7: "Person gesturing",
8: "Person digging",
9: "Person carrying an object",
10: "Person running",
11: "Person entering a facility",
12: "Person exiting a facility",
}

def get_org_event_type_by_id(id):
    pass
    return org_event_id_type_dict[id]


def get_obj_type_by_id(id):
    if not id in obj_id_type_dict.keys():
        return "Other"
    return obj_id_type_dict[id]


def get_event_type_by_id(event_id):
    return event_id_type_dict[event_id]


def get_object_id_list_from_one_hot(one_hot_labels):
    pass
    ret_list=[]
    for i in xrange(0,len(one_hot_labels)):
        if int(one_hot_labels[i])==1:
            ret_list.append(i)
    return ret_list
