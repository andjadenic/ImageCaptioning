# Image Captioning using ResNet and LSTM on MC COCO Dataset
### Goal
* The goal of this project is to enable machines to caption images automatically.
* For a given image that the machine sees for the first time as input, it should generate a sentence description of that picture as output.
* This work reproduces and analyzes the seminal paper: [**Show and Tell: A Neural Image Caption Generator (Vinyals et al., 2015)**](https://arxiv.org/abs/1411.4555).
<img width="1614" height="611" alt="image_captioning_task" src="https://github.com/user-attachments/assets/95a70463-f346-46f6-b1cd-b49ce489c38b" />

### Model architecture
The project follows an encoder–decoder structure:
* **Encoder**: A ResNet CNN extracts high-level image features.
* **Decoder**: An LSTM generates captions from these features.
Implementation is done in PyTorch.
<img width="1085" height="529" alt="encoder_decoder_image_captioning_example" src="https://github.com/user-attachments/assets/af3a51b1-8c67-4640-80e5-54d91a92fe5c" />

### Project Structure and Functionalities
The project is organized into modular steps reflecting the image captioning pipeline:
* `download_data.py` – scripts for downloading and preparing the MS COCO dataset.
* `preprocess.py` – dataset preprocessing (resizing, vocabulary construction, tokenization, word-to-index mappings).
* `model.py` – encoder (ResNet) and decoder (LSTM) model definitions.
* `train.py` – training loop with checkpointing, logging, and GPU support.
* `evaluate.py` – evaluation and inference on test images with BLEU score computation.
* `utils/` – helper functions for data loading, visualization, and caption formatting.

### Dataset
* Training data is [**MC COCO dataset**](https://cocodataset.org/).
* Includes 5 captions for 300k images.
* Preprocessing ensures consistent vocabulary and alignment between images and captions.
* 
### GPU Support
* This project utilizes **GPU acceleration** with PyTorcs.  
* Both the model and data tensors are automatically moved to the GPU if available.

### Requirements
This project requires the following Python packages:
* `torch`
* `torchvision`
* `pycocotools`
* `numpy`
* `matplotlib`
* `skimage`
* `nltk`
 
