# File with all hyperparameters and paths

# Paths
download_COCO_captions_train2017_json = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\download_COCO\captions_train2017.json'
train_data_path = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\train_data'
test_data_path = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\test_data'
csv_train_path = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\train_data\train_captions.csv'
train_annFile = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\annotations\captions_train2017.json'  # Path to json annotation file
test_annFile = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\annotations\captions_val2017.json'

# Hyperparameters
feature_size = 512  # size of a feature (context) vector that is encoder's output

# Text preprocessing
embed_size = feature_size
max_seq_length = 29077
L = max_seq_length

# LSTM
input_size = embed_size
hidden_size = 1
num_layers = 1

# Training
Nb = 32  # Batch size