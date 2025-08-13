from preprocess import *
from model import *
from config import *
import torch
torch.backends.cudnn.benchmark = True
import time


if __name__ == "__main__":
    vocabulary = Vocabulary()
    vocabulary.build_vocabulary(json_path=train_annFile)

    train_dataset = CocoDataset(data_path=train_data_path,
                                json_path=train_annFile,
                                vocabulary=vocabulary)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    encoder = EncoderCNN(feature_size=feature_size).to(device)
    decoder = DecoderRNN(embed_size=embed_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         vocabulary=vocabulary).to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    loss_track = []

    # how much time it takes to do a bached forward pass?
    start_time = time.time()
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
        #print(f'{epoch=}, {i=},  {loss=}')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        if i==1:
            break
    end_time = time.time()
    print(end_time - start_time)

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
