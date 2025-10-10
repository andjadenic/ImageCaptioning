import torch

# Paths
captions_train_path = r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\captions_train2017.json'
captions_val_path = r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\captions_val2017.json'
train_imgs_path = r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\train2017'
val_imgs_path = r'C:\Users\HP\AndjaDenic\ImageCaptioning\data\val2017'


encoder_path = r'trained_models/encoder.pth'
decoder_path = r'trained_models/decoder.pth'

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