import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import *
from utils import zero_after


class EncoderCNN(nn.Module):
    def __init__(self, feature_size):
        """Load the pretrained ResNet-152 and replace top fully connected layer
           so that ResNet's output has specific size of feature_size.
           Encoder = ResNet + fc linear layer + batch norm layer
        Args:
            feature_size: size of the encoder's output (feature) vector
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights='ResNet152_Weights.DEFAULT').to(device)

        resnet_layers = list(resnet.children())  # List of all high-level layers
        modules = resnet_layers[:-1]  # Delete the last fc layer
        self.resnet = nn.Sequential(*modules)  # ResNet slice of encoder

        self.linear = nn.Linear(2048, feature_size)  # Fc layer with (2048, ) input shape and (feature_size,) output shape
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)

        # Freeze the parameters of ResNet slice (resnet)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        """Extract feature vectors from input images"""
        features = self.resnet(images)  # size = (batch_size, 2048, 1, 1) tensor
        features = features.reshape(features.size(0), -1)  # size = (batch_size, 2048) tensor
        features = self.linear(features) # size = (batch_size, feature_size) tensor
        features = self.bn(features)  # size = (batch_size, feature_size) tensor
        #  Encoder's learnable parameters are linear layer's parameters and batch normalization's parameters.
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocabulary):
        """
        INPUTS:
        :param embed_size: word space size, this is an input size of LSTM block
                           embed_size is equal to feature_size because Encoder(image) and Embedding(word) belong to the same space
        :param hidden_size: size of hidden and cell states in LSTM block
        :param num_layers: number of stacked LSTM blocks
        """
        super(DecoderRNN, self).__init__()
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.L = vocabulary.L

        self.embed = nn.Embedding(self.vocab_size, embed_size)  # Embedding layer takes indexed sentence and outputs its embedding
                                                           # word_id -> ohe word_id -> We * ohe word_id
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)  # final fully connected layer

    def forward(self, feature_maps, input_captions, length_captions):
        """
        Forward pass of the decoder.

        Args:
            feature_maps (torch.Tensor): Image features from the encoder (batch_size, embedded_size).
            input_captions (torch.Tensor): Input captions (batch_size, caption_length).
            length_captions (torch.Tensor): Caption lengths (batch_size,).

        Returns:
            torch.Tensor: Predicted scores for each vocabulary word at each time step
                          (batch_size, caption_length, vocab_size).
        """
        embeddings = self.embed(input_captions)  # embedded representations of the current batch of input captions
                                                 # embeddings.shape = (batch_size, L, embed_size)
        # feature_maps.size = (Nb, embed_size)
        embeddings = torch.cat((feature_maps.unsqueeze(1), embeddings), 1)  # feature_maps are concatenated to embeddings
                                                                                        # embeddings.size = (Nb, L + 1, embed_size)
                                                                                        # embeddings = [feature_map, <start>, w1, w2, ..., wN, <pad>, ..., <pad>]
        # Both the image and the words are mapped to the same space, the image by using the encoder (ResNet + fcl + bnl),
        # the words by using word embedding We (fcl).
        # The image I is only input once, at t = −1, to inform the LSTM about the image contents threw input x_-1.
        # source: https://arxiv.org/pdf/1411.4555

        lengths = length_captions + 1  # after adding the image
        packed_input = pack_padded_sequence(embeddings, lengths, batch_first=True)  # removes padding and optimizes RNN processing

        # Pass through the LSTM
        # The hidden state and cell state are initialized to zeros by default if not provided.
        packed_h, _ = self.lstm(packed_input)
        h, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=self.L+1)  # h.shape = (Nb, L + 1, hidden_size)
            # h contains the hidden states (h_t) from the last layer of the LSTM, for each t.

        # Pass the LSTM outputs h through the linear layer to get vocabulary scores
        outputs = self.linear(h)  # outputs.size = (Nb, 1 + L, vocab_size)

        # We use output for the caption sequence, excluding the prediction
        # based on the initial image feature input alone.
        outputs = outputs[:, 1:, :]  # outputs.shape = (Nb, L, vocab_size)
        return outputs

    def sample(self, feature_maps):
        """Generate captions for given image features using greedy search.
        feature_maps (torch.Tensor): Image features from the encoder (batch_size, embedded_size).
        """
        captions_list = []
        states = (torch.zeros(num_layers, batch_size, hidden_size),
                  torch.zeros(num_layers, batch_size, hidden_size))
        lstm_input = feature_maps

        for i in range(self.L):
            # Forward feature map through LSTM blocks
            h, states = self.lstm(lstm_input, states)  # h.shape = (batch_size, hidden_size)

            # Forward hiddens through a linear layer to produce distribution over vocabulary words.
            curr_outputs = self.linear(h)  # outputs:  (batch_size, vocab_size)

            # Predicted word is the one with the highest probability
            curr_ids = curr_outputs.argmax(1)  # size: (batch_size,)
            captions_list.append(curr_ids)

            # Prepare input for the next word prediction
            # Next input is embedded current word
            lstm_input = self.embed(curr_ids)  # feature_maps: (batch_size, embed_size)

        # Output of this function is mini-batch of sequences cosisting of indexed words before <END> token
        output = torch.stack(captions_list, 1).to(torch.long)
        output = zero_after(output, self.vocabulary.end_idx)

        return output






