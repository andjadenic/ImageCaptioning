import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Vocabulary:
    def __init__(self, raw_sentences):
        self.word_to_id = {}
        self.id_to_word = {}
        self.unique_words = set()
        self.L = 0  # length of the longest sentence

        translator = str.maketrans('', '', string.punctuation)
        for raw_sentence in raw_sentences:
            sentence = raw_sentence.lower().translate(translator).split(' ')
            if len(sentence) > self.L:
                self.L = len(sentence)
            for word in sentence:
                self.unique_words.add(word)
        for id, word in enumerate(list(self.unique_words)):
            self.word_to_id[word] = id
            self.id_to_word[id] = word

class ToLowercase:
    def __call__(self, sentence):
        return sentence.lower()
class RemovePunctuation:
    def __call__(self, sentence):
        translator = str.maketrans('', '', string.punctuation)
        return sentence.translate(translator)
class Numericalize:
    def __init__(self, word_to_id):
        self.word_to_id = word_to_id
    def __call__(self, sentence):
        sentence_list = sentence.split(' ')
        return [self.word_to_id[word] for word in sentence_list]
class PadSequence:
    def __init__(self, L):
        self.L = L
    def __call__(self, sentence):
        return sentence + [0] * (self.L - len(sentence))

class TextPreprocessingPipeline:
    def __init__(self, vocab):
        self.word_to_id = vocab.word_to_id
        self.L = vocab.L
        self.numericalize = Numericalize(vocab.word_to_id)
        self.pad_sequence = PadSequence(vocab.L)


class SentenceDataset(Dataset):
    def __init__(self, sentences, sentiments, vocab, transform=None):
        self.raw_sentences = sentences
        self.sentiments = sentiments
        self.transform = transform

        self.word_to_id = vocab.word_to_id
        self.id_to_word = vocab.id_to_word
        self.L = vocab.L
        self.vocab_size = len(self.word_to_id)
    def __len__(self):
        return len(self.raw_sentences)
    def __getitem__(self, index):
        sentence = self.raw_sentences[index]
        sentiment = self.sentiments[index]

        length = len(RemovePunctuation()(ToLowercase()(sentence)).split(' '))
        processed_sentence = self.transform(sentence)
        processed_sentence = torch.tensor(processed_sentence, dtype=torch.long).to(device)

        return {
            'processed_sentence': processed_sentence,
            'label': torch.tensor(sentiment, dtype=torch.long).to(device),
            'length': torch.tensor(length).to(device)
        }


class TextProcessingPipeline:
    def __init__(self, vocab):
        self.word_to_id = vocab.word_to_id
        self.L = vocab.L
        self.numericalize = Numericalize(vocab.word_to_id)
        self.pad_sequence = PadSequence(vocab.L)
    def __call__(self, sentence_string):
        sentence_string = ToLowercase()(sentence_string)
        sentence_string = RemovePunctuation()(sentence_string)
        numerical_ids = self.numericalize(sentence_string)
        padded_ids = self.pad_sequence(numerical_ids)
        return padded_ids


def collate_fn(input_batch):
    input_batch.sort(key=lambda x: x['length'], reverse=True)

    sentences = [item['processed_sentence'] for item in input_batch]
    labels = [item['label'] for item in input_batch]
    lengths = [item['length'].item() for item in input_batch]

    return {
        'processed_sentence': torch.stack(sentences).to(torch.long),
        'label': torch.stack(labels).to(torch.long),
        'length': torch.tensor(lengths).to(torch.long)
    }


class model(nn.Module):
    def __init__(self, input_size):
        super(model, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    '''sentences = [
        "This movie is fantastic, i loved it!",  # 7
        "I hated every minute.",  # 4
        "truly captivating experience.",  # 3
        "I almost fell asleep.", # 4
        "Great acting and plot yeah." # 5
    ]
    sentiments = [1, 0, 1, 0, 1]

    vocab = Vocabulary(raw_sentences=sentences)

    text_transform = TextProcessingPipeline(vocab)

    sentence_dataset = SentenceDataset(sentences,
                                       sentiments,
                                       vocab=vocab,
                                       transform=text_transform)

    sentence_loader = DataLoader(
        sentence_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = model(input_size=len(vocab)).to(device)

    for i, batch in enumerate(sentence_loader):
        #print(f'{batch['processed_sentence']=}')
        #print(f'{batch['length']=}', '\n')

        packed_input = pack_padded_sequence(
            input=batch['processed_sentence'],
            lengths=batch['length'],
            batch_first=True
        )

        packed_output, _ = model(packed_input)

        print(f'{packed_output.data=}')
        print(f'{packed_output.batch_sizes=}')

        #unpacked_output, unpacked_length = pad_packed_sequence(
        #    sequence=packed_input,
        #    batch_first=True
        #)
        #print(f'{unpacked_output=}')
        #print(f'{unpacked_length=}', '\n')
        #print('\n')'''

    t = torch.tensor([1, 3, 2, 0]).to(torch.long)
    print(t + 1)