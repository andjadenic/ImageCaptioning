import torch
from pathlib import Path

# Paths

# Train Data
captions_train_path = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\captions_train2017.json')
train_data_path = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\train2017')
train_dataset_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\train_dataset.json')
train_dataset_1c_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\train_dataset_1c.json')
# Validation Data
captions_val_path = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\captions_val2017.json')
val_data_path = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\val2017')
val_dataset_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\val_dataset.json')
val_dataset_1c_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\val_dataset_1c.json')

# preprocessing
vocab_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\preprocess\vocab.json')


#encoder_path = r'trained_models/encoder.pth'
#decoder_path = r'trained_models/decoder.pth'

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_size = 512  # size of a feature (context) vector that is encoder's output
embed_size = feature_size  # Because both words (tokens) and images are embedded to the same vector space
num_layers = 1
hidden_size = 512

# Training
num_workers = 7
batch_size = 64
learning_rate = 1e-3
num_epochs = 1