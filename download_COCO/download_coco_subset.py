'''
Script downloads N_train random images from 2017 Train COCO dataset
and N_val random images from 2017 Val COCO dataset
and save their captions in separate files train_captions_path and val_captions_path
using Python API for COCO dataset pycocotools and json files:
download_COCO_captions_train2017_json and download_COCO_captions_val2017_json
downloaded from official COCO website: https://cocodataset.org/#download
'''

import os
from pycocotools.coco import COCO
import numpy as np
import csv
from config import *


# download N_train random images from 2017 train COCO dataset
path_ann_train = download_COCO_captions_train2017_json
dir_img_train = train_data_path

coco_train = COCO(path_ann_train)
list_of_all_ids = list(coco_train.imgs.keys())

# Create CSV file for train images captions
with open(csv_train_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img_name', 'caption1', 'caption2', 'caption3', 'caption4', 'caption5'])  # Write a header

# Download N_train images
N_train = 10000
rand_nums = np.random.randint(1, len(list_of_all_ids), N_train)  # generates N_train random numbers
for n in rand_nums:
    curr_id = list_of_all_ids[n]  # current id
    coco_train.download(dir_img_train, [curr_id])  # downloads image with given current id
    curr_img = coco_train.loadImgs([curr_id])[0]

    curr_name = curr_img['file_name']

    annIds = coco_train.getAnnIds(imgIds=curr_img['id'])
    curr_anns = coco_train.loadAnns(annIds)  # list of dictionaries containing all the captions for the current image
    curr_captions_list = []
    for i in range(5):
        curr_captions_list.append(curr_anns[i]['caption'].rstrip('. '))

    with open(csv_train_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([curr_name,
                         curr_captions_list[0],
                         curr_captions_list[1],
                         curr_captions_list[2],
                         curr_captions_list[3],
                         curr_captions_list[4]])

'''
# download N_val random images from 2017 Val COCO dataset
path_ann_val = f'download_COCO/captions_val2017.json'
dir_img_val = f'miniCOCO/val'

coco_val = COCO(path_ann_val)
list_of_all_ids_val = list(coco_val.imgs.keys())

# Create CSV file for validation images captions
with open('../miniCOCO/val_captions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img_name', 'caption1', 'caption2', 'caption3', 'caption4', 'caption5'])  # Write a header

# Download N_val images
N_val = 20
rand_nums = np.random.randint(1, len(list_of_all_ids_val), N_val)  # generates N_val random numbers
for n in rand_nums:
    curr_id = list_of_all_ids_val[n]  # current id
    coco_val.download(dir_img_val, [curr_id])  # downloads image with given current id
    curr_img = coco_val.loadImgs([curr_id])[0]

    curr_name = curr_img['file_name']

    annIds = coco_val.getAnnIds(imgIds=curr_img['id'])
    curr_anns = coco_val.loadAnns(annIds)  # list of dictionaries containing all the captions for the current image
    curr_captions_list = []
    for i in range(5):
        curr_captions_list.append(curr_anns[i]['caption'].rstrip('. '))

    with open('../miniCOCO/val_captions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([curr_name,
                         curr_captions_list[0],
                         curr_captions_list[1],
                         curr_captions_list[2],
                         curr_captions_list[3],
                         curr_captions_list[4]])
'''
