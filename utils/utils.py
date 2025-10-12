from skimage import io
import os
from matplotlib import pyplot as plt
import json
import torch


def display_image(img):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def make_dataset(ann_json, data_path, new_json_path):
    '''
    Function reads information about images and captions
    from ann_json and data_path
    and saves the dataset in new_json_path
    '''
    with open(ann_json, 'r') as file:
        data = json.load(file)

    dataset = []
    for img_info in data['images']:
        curr_img_name = img_info['file_name']
        curr_img_id = img_info['id']
        curr_img = 0
        #curr_img = io.imread(os.path.join(data_path, curr_img_name))
        curr_captions_list = []
        for ann in data['annotations']:
            if ann['image_id'] == curr_img_id:
                curr_captions_list.append(ann['caption'])

        curr_sample = {
            'img_id': curr_img_id,
            'img_name': curr_img_name,
            'img': curr_img,
            'captions': curr_captions_list
        }
        dataset.append(curr_sample)

    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f'Dataset is successfully created and saved in {new_json_path}.')

def zero_after(x: torch.Tensor, id: int) -> torch.Tensor:
    mask_id = (x == id)

    # Find the first index of id in each row (or m if not found)
    idx_id = torch.argmax(mask_id.int(), dim=1)
    has_id = mask_id.any(dim=1)

    # Create a mask for zeroing
    n, m = x.shape
    arange = torch.arange(m, device=x.device).expand(n, m)
    idx_id_expanded = idx_id.unsqueeze(1).expand_as(x)

    # Build the final mask
    zero_mask = (arange >= idx_id_expanded) & has_id.unsqueeze(1)

    # Zero out elements
    x = x.clone()  # avoid modifying input in-place
    x[zero_mask] = 0
    return x
