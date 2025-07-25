from config import train_annFile, train_data_path
import json
import os
from skimage import io


with open(train_annFile, 'r') as file:
    data = json.load(file)

captions_id_list = [item['id'] for item in data['annotations']]

id = 0  # sample_id
ann_id = captions_id_list[id]
for obj in data['annotations']:
    if obj['id'] == ann_id:
        img_id = obj['image_id']
        break
for obj in data['images']:
    if obj['id'] == img_id:
        img = io.imread(os.path.join(train_data_path, obj['file_name']))
        break

print(f'{id=}')
print(f'{ann_id=}')
print(f'{img_id=}')
print(f'{type(img)=}')
print(f'{img=}')