import os
import dataset_config as data_cfg

#   path to place365 pre-trained models
place_model_root_path="/mnt/hdd_8t/place365/"
category_name_fn=os.path.join(place_model_root_path,"categories_places365.txt")

#   model filename
model_fn=os.path.join(place_model_root_path,"resnet18_places365.pth.tar")


