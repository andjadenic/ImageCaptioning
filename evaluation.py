from model import *
from train import miniCOCO_vocabulary
import matplotlib.pyplot as plt
import skimage.io as io



# Define model (hyper)parameters
feature_size = 50
embed_size = feature_size  # Because both words (tokens) and images are embedded to the same vector space
hidden_size = 10
vocab_size = 924
max_seq_length = 27

# Load saved models
decoder_path = f'trained_models/decoder.pth'
encoder_path = f'trained_models/encoder.pth'

encoder = EncoderCNN(feature_size=feature_size)
decoder = DecoderRNN(embed_size=50,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                    max_seq_length=max_seq_length)

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

# Set the model to evaluation mode
encoder.eval()
decoder.eval()

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the center 224x224
    transforms.ToTensor(),  # Convert PIL image to a PyTorch tensor
    transforms.Normalize(  # Normalize the tensor with the ImageNet mean and standard deviation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Image for evaluation
img_path = f'miniCOCO/train/000000033272.jpg'
img = Image.open(img_path)

# Show image
I = io.imread(img_path)
plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(I)
plt.show()

# Forward pass
input = transform(img).unsqueeze(0)  # Add batch dimension
with torch.no_grad():
    features = encoder(input)

outputs, word_ids = decoder.sample(features)
print(f'{word_ids=}', '\n')
caption = []
for i in range(27):
    caption.append(miniCOCO_vocabulary.idx2word[word_ids[0, i].item()])
print(caption)

