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

    # Tokenize and Calculate Original Word Counts
    tokens = nltk.word_tokenize(processed_caption.lower())
    length = len(tokens)

    # Numericalize, Add Special Tokens (<START>, <END>)
    caption_indices = [vocab(token) for token in tokens]  # Maps tokens into indexes: [id(w1), id(w2), ..., id(wN)]
    full_indices = [vocab.start_idx] + caption_indices + [vocab.end_idx]   # Includes <START> and <END> indexes:
                                                         # [id(<START>), id(w1), id(w2), ..., id(wN), id(<END>)]

    # Calculate lengths for pack_padded_sequence (original length + 2)
    length_for_packing = length + 2

    # Prepare Input and Target Sequence (before padding)
    input_sequence = full_indices[:-1]  # [id(<START>), id(w1), id(w2), ..., id(wN)]
    target_sequence = full_indices[1:]  # [id(w1), id(w2), ..., id(wN), id(<END>)]

    # Pad Input and Target Sequences
    # Both input and target sequences for a given caption are padded to length N+1
    max_len_input_target = vocab.L + 1

    padded_inputs = []
    padded_targets = []

    # Pad input sequence to [id(<START>), id(w1), id(w2), ..., id(wN), id(<PAD>), ..., id(<PAD>)]
    pad_len_inp = max_len_input_target - len(input_sequence)
    padded_inp = input_sequence + [vocab.pad_idx] * pad_len_inp

    # Pad target sequence to  [id(w1), id(w2), ..., id(wN), id(<END>), id(<PAD>), ..., id(<PAD>)]
    pad_len_tgt = max_len_input_target - len(target_sequence)
    padded_tgt = target_sequence + [vocab.pad_idx] * pad_len_tgt

    # Convert to PyTorch Tensors
    captions_input_tensor = torch.tensor(padded_inp, dtype=torch.long)
    targets_tensor = torch.tensor(padded_tgt, dtype=torch.long)
    lengths_tensor = torch.tensor(length_for_packing, dtype=torch.long)  # Use the N+2 lengths

    return captions_input_tensor, lengths_tensor, targets_tensor


def target_transform(captions, vocabulary):
    '''Convert list of (tuple of) strings into torch tensors using pre-built vocabulary.'''

    if type(captions[0] == tuple):
        captions = captions[0]

    i, l, o = [], [], []
    for caption in captions:
        i_curr, l_curr, o_curr = preprocess_caption(caption, vocabulary)
        i.append(i_curr)
        l.append(l_curr)
        o.append(o_curr)
    return torch.stack(i), torch.stack(l), torch.stack(o)


if __name__ == '__main__':
    vocabulary = Vocabulary()
    vocabulary.build_vocabulary(json_path=train_annFile)

    captions = ['There is nothing...', '...left to DO.']

    input_captions, lengths, targets = target_transform(captions, vocabulary)

    print(f'{input_captions=}', '\n')
    print(f'{lengths=}', '\n')
    print(f'{targets=}', '\n')