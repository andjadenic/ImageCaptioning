import nltk
import torch
import json
import string
from config import train_annFile


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        self.pad_idx = self.add_word(self.pad_token)  # id(<PAD>) = 0
        self.start_idx = self.add_word(self.start_token)  # id(<START>) = 1
        self.end_idx = self.add_word(self.end_token)  # id(<END>) = 2
        self.unk_idx = self.add_word(self.unk_token)  # id(<UNK>) = 3

        self.L = 0  # Maximum length of a caption

    def add_word(self, word):
        '''Adds a word to the vocabulary
        :return: index of added word'''
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            return self.idx - 1  # Return the index assigned
        return self.word2idx[word]  # Return existing index

    def build_vocabulary(self, json_path):
        '''Builds the vocabulary from provided .json annotations'''
        print('Building vocabulary...')

        print(f"Loading annotations from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("Annotations loaded.")

        all_captions = [ann['caption'] for ann in data['annotations']]
        print(f"Extracted {len(all_captions)} captions.")

        # Preprocessing: Lowercase and remove punctuation
        print("Preprocessing captions (lowercasing and removing punctuation)...")
        translator = str.maketrans('', '', string.punctuation)
        processed_captions = (caption.lower().translate(translator) for caption in all_captions)

        # Tokenize captions and keep track of length of the longest caption
        tokenized_captions = []
        L = 0
        for caption in processed_captions:
            tokenized_caption = nltk.tokenize.word_tokenize(caption)
            tokenized_captions.append(tokenized_caption)
            if len(tokenized_caption) > L:
                L = len(tokenized_caption)
        self.L = L

        # Add all words to the vocabulary
        for caption in tokenized_captions:
            for token in caption:
                self.add_word(token)

        print(f'Vocabulary built with {len(self)} words.')

    def __call__(self, word):
        '''Looks up a word's index, returns <UNK> index if not found.'''
        return self.word2idx.get(word, self.unk_idx)

    def __len__(self):
        '''Returns the total size of the vocabulary.'''
        return len(self.word2idx)


def preprocess_caption(raw_caption, vocab):
    '''
    Preprocesses single raw caption for DecoderRNN

    :param raw_captions (str): Raw caption
    :param vocab: A Vocabulary object

    :return:
    tuple: A tuple containing:
        - captions_input_tensor (torch.Tensor): [id(<START>), id(w1), ..., id(wN)]
            Indexed and padded input sequence.
            Includes <START> index, excludes <END> index.
        - lengths_tensor (torch.Tensor): Original sequence lengths.
            Required by pack_padded_sequence for DecoderRNN's forward method.
            size = ([batch_size, ])
        - targets_tensor (torch.Tensor):  [id(w1), ..., id(wN), id(<END>)]
            Indexed and padded target sequences for loss calculation.
            Excludes <START>, includes <END>.
    '''

    # Remove punctuation, lower case.
    translator = str.maketrans('', '', string.punctuation)
    processed_caption = raw_caption.lower().translate(translator)

    tokens = nltk.word_tokenize(processed_caption.lower())

    # Numericalize, Add Special Tokens (<START>, <END>)
    caption_indices = [vocab(token) for token in tokens]  # Maps tokens into indexes: [id(w1), id(w2), ..., id(wN)]
    full_indices = [vocab.start_idx] + caption_indices + [vocab.end_idx]   # Includes <START> and <END> indexes:
                                                         # [id(<START>), id(w1), id(w2), ..., id(wN), id(<END>)]

    # Prepare Input and Target Sequence (before padding)
    input_sequence = full_indices[:-1]  # [id(<START>), id(w1), id(w2), ..., id(wN)]
    target_sequence = full_indices[1:]  # [id(w1), id(w2), ..., id(wN), id(<END>)]

    # Pad Input and Target Sequences

    # Pad input sequence to [id(<START>), id(w1), id(w2), ..., id(wN), id(<PAD>), ..., id(<PAD>)]
    padded_input = input_sequence + [vocab.pad_idx] * (vocab.L - len(input_sequence))

    # Pad target sequence to  [id(w1), id(w2), ..., id(wN), id(<END>), id(<PAD>), ..., id(<PAD>)]
    padded_target = target_sequence + [vocab.pad_idx] * (vocab.L - len(target_sequence))

    # Convert to PyTorch Tensors
    captions_input_tensor = torch.tensor(padded_input, dtype=torch.long)
    targets_tensor = torch.tensor(padded_target, dtype=torch.long)
    lengths_tensor = torch.tensor(len(tokens), dtype=torch.long)

    return captions_input_tensor, lengths_tensor, targets_tensor


def target_transform(captions, vocabulary):
    '''Convert list of (tuple of) strings into torch tensors using pre-built vocabulary.'''
    captions = captions[0]

    i, l, o = [], [], []
    for caption in captions:
        i_curr, l_curr, o_curr = preprocess_caption(caption, vocabulary)
        i.append(i_curr)
        l.append(l_curr)
        o.append(o_curr)
        print(f'{i_curr.shape=}')
        print(f'{l_curr.shape=}')
        print(f'{o_curr.shape=}')
    return torch.stack(i), torch.stack(l), torch.stack(o)


if __name__ == '__main__':
    vocabulary = Vocabulary()
    vocabulary.build_vocabulary(json_path=train_annFile)

    captions = [('There is nothing...', '...left to DO.')]

    input_captions, lengths, targets = target_transform(captions, vocabulary)

    print(f'{input_captions=}', '\n')
    print(f'{input_captions.shape=}', '\n')
    print(f'{lengths=}', '\n')
    print(f'{lengths.shape=}', '\n')
    print(f'{targets=}', '\n')
    print(f'{targets.shape=}', '\n')