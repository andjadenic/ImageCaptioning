from model.models import *
from utils.config import *
import torch
import time
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Load datasets
    train_dataset_1c = read_dataset(train_dataset_1c_json)
    val_dataset_1c = read_dataset(val_dataset_1c_json)[:12500]

    # Make vocabulary
    vocabulary = Vocabulary()
    vocabulary.load_vocab(vocab_json)
    vocab_size = len(vocabulary)

    # Define Dataset and DataLoader
    train_dataset_1c = CocoDataset1c(data_path=train_data_path, dataset_1c_path=train_dataset_1c_json,
                                     vocabulary=vocabulary,
                                     transform=transform_img, target_transform=preprocess_caption)
    train_loader_1c = DataLoader(train_dataset_1c,
                                 batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, collate_fn=collate_fn_1d, pin_memory=True)
    val_dataset_1c = CocoDataset1c(data_path=val_data_path, dataset_1c_path=val_dataset_1c_json,
                                   vocabulary=vocabulary,
                                   transform=transform_img, target_transform=preprocess_caption)
    val_loader_1c = DataLoader(val_dataset_1c,
                               batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=collate_fn_1d, pin_memory=True)


    # Define models
    encoder = EncoderCNN(feature_size=feature_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, num_layers, vocabulary).to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # Calculating the loss of variable-length sequences: https://www.codegenes.net/blog/packed-sequence-into-loss-pytorch/
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    training_info = {
        'hidden_size': hidden_size,
        'batch_size': batch_size,
        'lr': learning_rate,
        'epoch_avg_train_loss': [],
        'epoch_avg_val_loss': [],
        'time': 0
    }

    print('Training starts.')

    start_time = time.time()  # Start timer

    for epoch in range(1, num_epochs + 1):
        #  ---- Training Phase ----
        encoder.train()
        decoder.train()
        curr_epoch_avg_loss_track = 0.0
        num_batches = 0

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
            curr_epoch_avg_loss_track += batch_loss.item()
            num_batches += 1

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        avg_train_loss = curr_epoch_avg_loss_track / num_batches
        training_info['epoch_avg_train_loss'].append(avg_train_loss)

        # ---- Evaluation Phase ----
        encoder.eval()
        decoder.eval()
        curr_epoch_avg_loss_track = 0.0
        num_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader_1c):
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
                curr_epoch_avg_loss_track += batch_loss.item()
                num_batches += 1
            avg_val_loss = curr_epoch_avg_loss_track / num_batches
            training_info['epoch_avg_val_loss'].append(avg_val_loss)
        print(
            f"Epoch {epoch}  "
            f"Train Loss: {avg_train_loss:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}  "
            "\n"
        )

    end_time = time.time()
    training_time = end_time - start_time
    print(f'Model training has been successfully completed in {training_time / 60:.2f} minutes.')
    training_info['training_time'] = training_time

    # After training loop finishes:
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")

    # Save training info parameters
    with open('training_info.json', 'w') as json_file:
        json.dump(training_info, json_file, indent=4)

    # 1. Instantiate model objects with the same hyperparameters:
    #encoder = EncoderCNN(feature_size=embed_size).to(device)
    #decoder = DecoderRNN(embed_size, hidden_size, num_layers, vocabulary).to(device)

    # 2. Load saved states:
    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

    # 3. Set to evaluation or training mode as needed:
    encoder.eval()
    decoder.eval()
