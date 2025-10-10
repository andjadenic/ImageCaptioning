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