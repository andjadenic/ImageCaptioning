# Image Captioning using ResNet and LSTM on MC COCO Dataset
## Project description
The goal of this project is to enable machine to **caption images automatically**, based on a given dataset of image-sentence pairs. The project uses a **supervised machine learning** paradigm.
* For a given image that the machine sees for the first time as input, it should generate a sentence description of that picture as output.
<img width="1614" height="611" alt="image_captioning_task" src="https://github.com/user-attachments/assets/95a70463-f346-46f6-b1cd-b49ce489c38b" />

## Dataset
The [COCO (Common Objects in Context)](https://cocodataset.org/#download) dataset is a large-scale image recognition dataset used for various computer vision tasks like object detection, segmentation, and captioning. COCO is benchmark dataset commonly used in machine learning—both for research and practical applications.

Overview of COCO 2017 dataset is given in [this blogpost](https://www.v7labs.com/blog/coco-dataset-guide) and in [paper 'Microsoft COCO: Common Objects in Context'](https://arxiv.org/abs/1405.0312).

Dataset contains over **123,000 images, each annotated with 5 captions describing the scene**, splitted into train (118,000) and validation (5,000) subsets that can be downloaded via [COCO website](https://cocodataset.org/#download).


## Model architecture
This work uses architecture described in paper: [**Show and Tell: A Neural Image Caption Generator (Vinyals et al., 2015)**](https://arxiv.org/abs/1411.4555), that follows an encoder–decoder structure:
* **Encoder**: A ResNet CNN extracts high-level image features.
* **Decoder**: An LSTM generates captions from these features.

<img width="1085" height="529" alt="encoder_decoder_image_captioning_example" src="https://github.com/user-attachments/assets/af3a51b1-8c67-4640-80e5-54d91a92fe5c" />

## Project Structure and Functionalities
The project is organized into modular steps reflecting the image captioning pipeline:
* `download_COCO.py` – scripts for downloading and preparing the MS COCO dataset.
* `coco_dataset_overview` - exploratory data analysis notebook.
* `preprocess.py` – dataset preprocessing (vocabulary construction, tokenization, word-to-index mappings, COCO dataset).
* `model.py` – encoder (ResNet) and decoder (LSTM) model definitions.
* `train.py` – training loop with checkpointing, logging, and GPU support.
* `evaluate.py` – evaluation and inference on test images with BLEU score computation.
* `utils/` – helper functions for defining hyperparameters, visualizations, and additional functions.

## GPU Support
* This project utilizes **GPU acceleration** with PyTorcs.  
* Both the model and data tensors are automatically moved to the GPU if available.

## Requirements
This project requires the following Python packages:
* `torch`
* `torchvision`
* `pycocotools`
* `numpy`
* `matplotlib`
* `skimage`
* `nltk`
 
