### Goal
* The goal of this project is to enable machines to caption images automatically.
* For a given image that the machine sees for the first time as input, it should generate a sentence description of that picture as output.
<img width="1614" height="611" alt="image_captioning_task" src="https://github.com/user-attachments/assets/95a70463-f346-46f6-b1cd-b49ce489c38b" />

### Model architecture
* Model architecture is encoder (ResNet) - decoder (LSTM).
* The primary objective of this work is to reproduce and analyze the seminal ["Show and Tell: A Neural Image Caption Generator" (Vinyals et al., 2015)](https://arxiv.org/abs/1411.4555).
* Model is implemented using '''PyTorch'''.
<img width="1085" height="529" alt="encoder_decoder_image_captioning_example" src="https://github.com/user-attachments/assets/af3a51b1-8c67-4640-80e5-54d91a92fe5c" />

### Dataset
Data used for training the model is the [MC COCO dataset](https://cocodataset.org/).

### GPU Support
* This project utilizes GPU acceleration via '''PyTorch'''.  
* Both the model and data tensors are automatically moved to the GPU if available.


 
