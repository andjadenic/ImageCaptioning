from preprocess.preprocess import *
from model.models import *
from utils.config import *
import torch
torch.backends.cudnn.benchmark = True
import time


if __name__ == "__main__":
    # Load datasets
    train_dataset = read_dataset(train_dataset_json)
    train_dataset_1c = read_dataset(train_dataset_1c_json)

    # Make vocabulary
    vocabulary = Vocabulary()
    vocabulary.load_vocab(vocab_json)
    vocab_size = len(vocabulary)

    # Define Dataset and DataLoader
    train_dataset_1c = CocoDataset1c(data_path=train_data_path,
                                     dataset_1c_path=train_dataset_1c_json,
                                     vocabulary=vocabulary)
    train_loader_1c = DataLoader(
        train_dataset_1c,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_1d,
        pin_memory=True
    )

    encoder = EncoderCNN(feature_size=feature_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, num_layers, vocabulary).to(device)


    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # Calculating the loss of variable-length sequences: https://www.codegenes.net/blog/packed-sequence-into-loss-pytorch/
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    train_epoch_loss_track = []

    start_time = time.time()
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader_1c):
            img = batch['img_tensor'].to(device)  # (Nb, 3, 224, 224)
            cap = batch['cap_tensor'].to(device)  # (Nb, L+1)
            length = batch['length_tensor']  # (Nb, )

            # Forward pass
            features = encoder(img)
            padded_outputs, outputs_lengths = decoder(features, cap, length)

            # Define padded targets: [id( < w1 >), id( < w2 >), ..., id( < w_l >), id(<END>), id(<PAD>)...]
            padded_targets = cap[:, 1:]  # (Nb, L+1)

            # Calculating the loss
            mask = torch.zeros_like(padded_targets, dtype=torch.bool)  # (Nb, L+1)
            for n_batch, length in enumerate(outputs_lengths):
                mask[n_batch, :length] = True

            loss = criterion(padded_outputs.reshape(-1, vocab_size), padded_targets.reshape(-1))
            loss = loss.reshape(padded_targets.shape)
            masked_loss = mask * loss
            batch_loss = masked_loss.sum() / mask.sum()

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if i==1:
                break

    end_time = time.time()
    print(f'{end_time - start_time = }')

'''
    # Train the model
    for epoch in range(num_epochs):
        encoder.train(True)
        decoder.train(True)
        for i, batch in enumerate(train_loader):
            img_tensor = batch['img_tensor'].to(device)
            input_tensor = batch['input_tensor'].to(device)
            length_tensor = batch['length_tensor']
            target_tensor = batch['target_tensor'].to(device)

            # Forward pass
            feature_maps = encoder(img_tensor)
            outputs = decoder(feature_maps, input_tensor, length_tensor)

            # Calculating the loss
            loss = criterion(outputs.reshape(-1, len(vocabulary)), target_tensor.reshape(-1))
            print(f'{epoch=}, {i=},  {loss=}')

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation loss
            # model.eval()
            # https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

        loss_track.append(loss)

#        torch.save(encoder.state_dict(), encoder_path)
#        torch.save(decoder.state_dict(), decoder_path)
'''

    # 260s 2 forward passes with num_workers=4
    # s 2 forward+backward passes with num_workers=4
