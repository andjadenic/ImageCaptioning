# File with all hyperparameters and paths

# Paths
download_COCO_captions_train2017_json = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\download_COCO\captions_train2017.json'

train_data_path = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\train_data'
test_data_path = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\test_data'

train_annFile = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\annotations\captions_train2017.json'  # Path to json annotation file
test_annFile = r'C:\Users\csuser\Documents\andja_denic\ImageCaptioning\data\annotations\captions_val2017.json'

# Hyperparameters
feature_size = 512  # size of a feature (context) vector that is encoder's output
embed_size = feature_size  # Because both words (tokens) and images are embedded to the same vector space
num_layers = 2
hidden_size = 256

# Training
num_workers = 0
batch_size = 2
learning_rate = 0.01
num_epochs = 40