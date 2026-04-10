# Image Captioning using ResNet and LSTM on MC COCO Dataset
This is the project I did for my master theses.
* [Master theses](https://github.com/andjadenic/ImageCaptioning/blob/main/Master_rad.pdf)
* [YouTube video](https://youtu.be/bfHZcrYQxHU?si=UmBoK0rrxRMaRrIK) of defending master:

[![Watch the video](https://img.youtube.com/vi/bfHZcrYQxHU/hqdefault.jpg)](https://www.youtube.com/watch?v=bfHZcrYQxHU)

## Project description
The goal of this project is to enable machine to **caption images automatically**, based on a given dataset of image-sentence pairs. The project uses a **supervised machine learning** paradigm.
* For a given image that the machine sees for the first time as input, it should generate a sentence description of that picture as output.
<img width="1614" height="611" alt="image_captioning_task" src="https://github.com/user-attachments/assets/95a70463-f346-46f6-b1cd-b49ce489c38b" />

## Dataset
The [COCO (Common Objects in Context)](https://cocodataset.org/#download) dataset is a large-scale image recognition dataset used for tasks such as object detection, segmentation, and captioning. COCO is a benchmark dataset commonly used in machine learning, both for research and practical applications.

Exploratory Data Analysis (EDA) of the MC COCO 2017 dataset is given in [EDA.ipynb](https://github.com/andjadenic/ImageCaptioning/blob/main/EDA/EDA.ipynb).

The dataset contains over **$123,000$ images, each annotated with $5$ captions describing the scene**, splitted into train ($118,000$) and validation ($5,000$) subsets that can be downloaded via [COCO website](https://cocodataset.org/#download).

* Height and width of train image varies from $51$ (min) to $640$ (max) and from $59$ (min) to $640$ (max) pixels.
* $99.7%$ of images have 5 captions, and the rest of them have $6$ and $7$ captions.
* Vocabulary built out of captions from the training dataset has $29,077$ tokens (words).
* Number of words per caption varies from $5$ to $49$ with median of $10$.

### Data Preprocessing
**Image preprocessing** pipeline prepares the image for the encoder:
1. Resizing so that the shortest side is $256$ pixels.
2. Cropping a $224 \times 224$ patch from the center of the resized image.
3. Converting the image into a PyTorch tensor.
4. Scaling pixel values from $[0, 255]$ into $[0.0, 1.0]$.
5. Normalizing the tensor image channel-wise so the mean is $[0.485, 0.456, 0.406]$ and the standard deviation is $[0.229, 0.224, 0.225]$.

**Caption preprocessing** pipeline prepares caption for decoder:
1. Removal of punctuation marks.
2. All letters are converted to lowercase.
3. The description is tokenized. Each caption (string) is converted into a list of words (strings).
4. Words are indexed. Each description becomes a list of indices.
5. Special tokens indicating the beginning ($1$) and end ($2$) are added. Each caption becomes a list of indices starting with $1$ and ending with $2$.
6. Descriptions are padded with special tokens ($0$)) so that each description has the same length $L$.

Preprocessing pipelines are defined in [preprocess/preprocess.py](https://github.com/andjadenic/ImageCaptioning/blob/main/preprocess/preprocess.py).

The dataset is split into $118,000$ samples for training, $2,500$ for validation, and $2,500$ for testing.

## Model architecture
This work uses the architecture described in the paper: [**Show and Tell: A Neural Image Caption Generator (Vinyals et al., 2015)**](https://arxiv.org/abs/1411.4555), which follows an encoder–decoder structure:
* **Encoder**: A ResNet CNN extracts high-level image features.
* **Decoder**: An LSTM generates captions from these features.

<img width="1085" height="529" alt="encoder_decoder_image_captioning_example" src="https://github.com/user-attachments/assets/af3a51b1-8c67-4640-80e5-54d91a92fe5c" />

## Model Parameters
**Encoder** (ResNet-based):
* Frozen: All pre-trained ResNet convolutional layers.
* Learnable: The classification linear layer and all Batch Normalization parameters.

**Decoder** (LSTM-based):
  * Learnable: All embedding (linear) layer parameters and LSTM cell parameters.

## Training
Training is done in $10$ epochs using gradient descent and Adam optimizer with a learning rate $10^{−3}$. The batch size was
$64$.
In each epoch, the average loss function is calculated on the training and validation datasets. The model with the smallest epoch average loss function on the validation dataset is saved. The saved model is used for inference and evaluation.
The model was trained on a computer with an 8 GB NVIDIA Quadro RTX 4000 graphics card, a processor with 8 cores and 16 threads, and 64GB of RAM.

## Evaluation
The trained model is evaluated on the test dataset. Achived evaluation scores are obtained on test dataset:

* BLEU-1 = 68.36
* BLEU-2 = 50.35
* BLEU-3 = 35.70
* BLEU-4 = 25.23
* ROUGE-L = 49.51
* CIDEr = 82.62 

## GPU Support
* This project utilizes **GPU acceleration** with PyTorcs.  
* Both the model and data tensors are automatically moved to the GPU if available.

## Requirements
This project requires the following Python packages:
* `torch`
* `torchvision`
* `PIL`
* `sckit-image`
* `statistics`
* `pycocoevalcap`
* `matplotlib` 
