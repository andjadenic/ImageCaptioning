import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms
from PIL import Image


class EncoderCNN(nn.Module):
    def __init__(self, feature_size):
        """Load the pretrained ResNet-152 and replace top fc layer
        Args:
            feature_size: size of the output feature map
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights='ResNet152_Weights.DEFAULT')

        resnet_layers = list(resnet.children())  # List of all high-level layers
        modules = resnet_layers[:-1]  # Delete the last fc layer
        self.resnet = nn.Sequential(*modules)  # ResNet slice of encoder

        self.linear = nn.Linear(resnet.fc.in_features, feature_size)  # Fc layer with (2048,) input shape and (feature_size,) output shape
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)

        # Freeze the parameters of ResNet slice (resnet)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)  # size = (batch_size, 2048, 1, 1) tensor
        features = features.reshape(features.size(0), -1)  # size = (batch_size, 2048) tensor
        features = self.bn(self.linear(features))  # size = (batch_size, feature_size) tensor
        #  Encoder's learnable parameters are linear layer's parameters and batch normalization's parameters.
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=30):
        """
        INPUTS:
        :param embed_size: word space dimension, this is an input size of LSTM block
        :param hidden_size: size of hidden and cell states
        :param vocab_size: number of words in vocabulary
        :param num_layers: number of stacked LSTM blocks
        :param max_seq_length: maximum length of a sequence
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding layer takes indexed sentence and outputs its embedding
                                                           # word_id -> ohe word_id -> We * ohe word_id
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)  # final fully connected layer

        self.max_seg_length = max_seq_length

    def forward(self, feature_maps, input_captions):
        """
        Forward pass of the decoder.

        Args:
            feature_maps (torch.Tensor): Image features from the encoder (batch_size, embedded_size).
            input_captions (torch.Tensor): Input captions (batch_size, caption_length).

        Returns:
            torch.Tensor: Predicted scores for each vocabulary word at each time step
                          (batch_size, caption_length, vocab_size).
        """
        embeddings = self.embed(input_captions)  # embedded representations of the current batch of input captions
                                                 # embeddings.shape = (batch_size, max_seq_len, embed_size)
        # featute_maps.size = (Nb, embed_size)
        embeddings = torch.cat((feature_maps.unsqueeze(1), embeddings), 1)  # feature_maps are concatenated to embeddings
                                                                                        # embeddings.size = (Nb, max_seq_len + 1, embed_size)
                                                                                        # embedding = [feature_map, <start>, w1, w2, ..., wN, <pad>, ..., <pad>]
        # Both the image and the words are mapped to the same space, the image by using the encoder (CNN, ResNet + fcl),
        # the words by using word embedding We (fcl).
        # The image I is only input once, at t = −1, to inform the LSTM about the image contents threw input x_-1.
        # source: https://arxiv.org/pdf/1411.4555

        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # removes padding and optimizes RNN processing

        # Pass through the LSTM
        # The hidden state and cell state are initialized to zeros by default if not provided.
        h, _ = self.lstm(embeddings)  # h.shape = (Nb, 1 + max_seq_length, hidden_size)

        # Pass the LSTM outputs h through the linear layer to get vocabulary scores
        outputs = self.linear(h)  # outputs.size = (Nb, 1 + max_seq_length, vocab_size)

        # We use output for the caption sequence, excluding the prediction
        # based on the initial image feature input alone.
        outputs = outputs[:, 1:, :]  # outputs.shape = (Nb, max_seq_length, vocab_size)
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        outputs_all = []
        inputs = features.unsqueeze(1)  # Add batch dimension
                                        # inputs.shape = (batch_size, 1, embed_size)
        for i in range(self.max_seg_length):
            # Forward feature map through LSTM blocks
            hiddens, states = self.lstm(inputs, states)  # hiddens.shape = (batch_size, 1, hidden_size)

            # Output of the LSTM blocks is hiddens.
            # Forward hiddens through a linear layer to produce distribution over vocabulary words.
            outputs_i = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)

            # Predicted word is the one with the highest probability
            _, predicted_id_words = outputs_i.max(1)  # predicted_id_words: (batch_size)
            sampled_ids.append(predicted_id_words)
            outputs_all.append(outputs_i)

            # Prepare input for the next word prediction.
            # Next input is embedded current word
            with torch.no_grad():
                inputs = self.embed(predicted_id_words)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)

        # Output of this function is mini-batch of sequences cosisting of indexed words
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        outputs_all = torch.stack(outputs_all, 1)
        return outputs_all, sampled_ids


if __name__ == '__main__':
    # Define hyper parameters
    features_size = 100  # size of Encoder output (feature map)

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

    img_path = f'miniCOCO/train/000000000438.jpg'
    img = Image.open(img_path)
    input = transform(img).unsqueeze(0)  # Add batch dimension

    model1 = EncoderCNN(100)
    model1.eval()
    features = model1(input)
    print(features.shape)  # (1, 100)



