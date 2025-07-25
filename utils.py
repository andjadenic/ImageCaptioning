from skimage import io
import os
from matplotlib import pyplot as plt
import json


def display_image(img):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def get_ith_sample(id, annFile, data_path):
    with open(annFile, 'r') as file:
        data = json.load(file)

    captions_id_list = [item['id'] for item in data['annotations']]

    ann_id = captions_id_list[id]
    for obj in data['annotations']:
        if obj['id'] == ann_id:
            img_id = obj['image_id']
            caption = obj['caption']
            break
    for obj in data['images']:
        if obj['id'] == img_id:
            img = io.imread(os.path.join(data_path, obj['file_name']))
            break
    return {
        'img_id': img_id,
        'img': img,
        'ann_id': ann_id,
        'caption': caption
    }

