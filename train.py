import nltk
import csv
from data.preprocess_captions import Vocabulary, preprocess_caption_for_decoder
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import EncoderCNN, DecoderRNN


class miniCOCODataset(Dataset):
    def __init__(self, csv_file, root_dir, vocabulary):
        """
        miniCOCODataset uses every image in data to make 5 samples,
        as there are 5 captions for each image.
        :param csv_file (str): Path to the csv file with captions
        :param root_dir (str): Directory with all the images
        :param vocabulary object
        """
        self.miniCOCO_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array from io.imread to PIL Image
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.vocabulary = vocabulary

    def __len__(self):
        return 5 * len(self.miniCOCO_frame)  # each image has 5 captions

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get raw image and apply transforms
        idx_name = idx // 5
        img_name = self.miniCOCO_frame.iloc[idx_name]['img_name']
        img_path = self.root_dir + img_name
        image = io.imread(img_path)  # ndarray image
        transformed_image = self.transform(image)  # Apply transforms

        # Get raw caption
        idx_caption = idx % 5 + 1
        caption = self.miniCOCO_frame.iloc[idx_name]['caption'+str(idx_caption)]

        # Out of caption Make input and target tensors for training decoder
        input_caption, length, target = preprocess_caption_for_decoder(caption, self.vocabulary)

        return transformed_image, input_caption, length, target




if __name__ == "__main__":
    # Make train miniCOCO Dataset
    train_miniCOCO_dataset = miniCOCODataset(csv_file=csv_train_path,
                                             root_dir=train_root_dir,
                                             vocabulary=miniCOCO_vocabulary)

    # Define model hyperparameters
    feature_size = 50
    embed_size = feature_size  # Because both words (tokens) and images are embedded to the same vector space
    hidden_size = 10
    num_layers = 1
    max_seq_length = miniCOCO_vocabulary.max_caption_len + 2

    learning_rate = 0.01
    num_epochs = 40
    batch_size = 8

    # Make train miniCOCO DataLoader
    miniCOCO_dataloader = DataLoader(train_miniCOCO_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=4)

    vocab_size = len(miniCOCO_vocabulary)
    print(f'{vocab_size=}')

    loss_track = []
    # Build the models
    encoder = EncoderCNN(feature_size=feature_size)
    decoder = DecoderRNN(embed_size=50,
                         hidden_size=hidden_size,
                         vocab_size=vocab_size,
                         num_layers=num_layers,
                         max_seq_length=max_seq_length)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=miniCOCO_vocabulary.pad_idx)  # Computes the cross entropy loss
                                                                # between input logits (outputs of decoder) and target
                                                                # one word at a time
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)


    # Train the models
    total_step = len(train_miniCOCO_dataset)
    for epoch in range(num_epochs):
        for i, (images, input_captions, lengths, targets) in enumerate(miniCOCO_dataloader):

            # Forward pass
            feature_maps = encoder(images)
            outputs = decoder(feature_maps, input_captions)
            #  RuntimeError: Expected target size [8, 924], got [8, 27]

            # Calculating the loss
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            print(f'{epoch=},  {loss=}')
            loss_track.append(loss)

            # Backward pass
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(encoder.state_dict(), "trained_models/encoder.pth")
    torch.save(decoder.state_dict(), "trained_models/decoder.pth")
