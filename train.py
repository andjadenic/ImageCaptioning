from preprocess import target_transform
import torch
from model import EncoderCNN, DecoderRNN
from preprocess import Vocabulary, preprocess_caption
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import *


if __name__ == "__main__":

    # Make a vocabulary
    vocabulary = Vocabulary()
    vocabulary.build_vocabulary(json_path=train_annFile)

    # Make train Dataset and DataLoader
    # Define preprocessing transformations for the images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Built-in CocoCaptions class to load the train dataset
    train_data = datasets.CocoCaptions(
        root=train_data_path,
        annFile=train_annFile,
        transform=transform  # preprocess images
    )

    # DataLoader for batching and shuffling
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)  # num_workers determines how many subprocesses to use for data loading

    loss_track = []
    # Build the models
    encoder = EncoderCNN(feature_size=feature_size)
    decoder = DecoderRNN(embed_size=embed_size,
                         hidden_size=hidden_size,
                         vocab_size=len(vocabulary),
                         num_layers=num_layers,
                         max_seq_length=vocabulary.L)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)  # Computes the cross entropy loss
                                                        # between input logits (outputs of decoder) and target
                                                        # one word at a time
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the model
    total_step = len(train_data)
    for epoch in range(num_epochs):
        for i, (images, captions) in enumerate(train_loader):
            # Preprocess captions
            input_captions, lengths, targets = target_transform(captions, vocabulary)


            # Forward pass
            feature_maps = encoder(images)
            outputs = decoder(feature_maps, input_captions)

            # Calculating the loss
            loss = criterion(outputs.reshape(-1, len(vocabulary)), targets.reshape(-1))
            print(f'{epoch=},  {loss=}')
            loss_track.append(loss)

            # Backward pass
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(encoder.state_dict(), "trained_models/encoder.pth")
    torch.save(decoder.state_dict(), "trained_models/decoder.pth")