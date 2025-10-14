import torch
from pathlib import Path

# Paths

# data
captions_train_path = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\captions_train2017.json')
train_data_path = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\train2017')
train_dataset_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\train_dataset.json')
train_dataset_1c_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\train_dataset_1c.json')

# preprocessing
vocab_json = Path(r'C:\Users\HP\AndjaDenic\ImageCaptioning\preprocess\vocab.json')


#encoder_path = r'trained_models/encoder.pth'
#decoder_path = r'trained_models/decoder.pth'

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_size = 512  # size of a feature (context) vector that is encoder's output
embed_size = feature_size  # Because both words (tokens) and images are embedded to the same vector space
num_layers = 1
hidden_size = 256

# Training
num_workers = 8
batch_size = 512
learning_rate = 5e-5
num_epochs = 10