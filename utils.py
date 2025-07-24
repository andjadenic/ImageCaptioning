from skimage import io
import os
from matplotlib import pyplot as plt
import json


def get_ith_image(data_path, i):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_files = [f for f in os.listdir(data_path)
                   if f.lower().endswith(supported_extensions)]
    image_files.sort()

    if i < 0 or i >= len(image_files):
        raise IndexError(f"Index {i} is out of bounds for {len(image_files)} image(s).")

    img_path = os.path.join(data_path, image_files[i])
    img = io.imread(img_path)
    return {
        'img_name': image_files[i],
        'img': img
    }


def display_image(img):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def get_captions(annFile, img_name):
    '''
    Collect captions for a given image name from the COCO 2017 annotation JSON file.
    '''
    with open(annFile, 'r') as f:
        coco_data = json.load(f)

    # find the image ID
    for image_info in coco_data.get('images', []):
        if image_info.get('file_name') == img_name:
            image_id = image_info['id']
            break
    if image_id is None:
        print(f'{img_name} not found in {annFile}')

    # collect image captions
    captions = []
    for annotation in coco_data.get('annotations', []):
        if annotation.get('image_id') == image_id:
            captions.append(annotation.get('caption'))

    return captions