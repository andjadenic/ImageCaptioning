import nltk
import torch
import csv
from config import csv_train_path


class Vocabulary:
    def __init__(self, freq_threshold=1):
        '''Initialize mappings and special tokens
        :param freq_threshold: only add words appearing >= freq_threshold'''
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        self.pad_idx = self.add_word(self.pad_token)  # Usually 0
        self.start_idx = self.add_word(self.start_token)  # Usually 1
        self.end_idx = self.add_word(self.end_token)  # Usually 2
        self.unk_idx = self.add_word(self.unk_token)  # Usually 3

        self.freq_threshold = freq_threshold
        self.word_freq = {}  # Frequency of each word in vocabulary in dataset

        self.max_caption_len = 0

    def add_word(self, word):
        '''Adds a word to the vocabulary
        :return: index of added word'''
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            return self.idx - 1  # Return the index assigned
        return self.word2idx[word]  # Return existing index

    def build_vocabulary(self, sentences_list):
        '''Builds the vocabulary from a list of tokenized sentences'''
        print('Building vocabulary...')
        for curr_sequence in sentences_list:

            # Keep track of the length of the longest caption
            if len(curr_sequence) > self.max_caption_len:
                self.max_caption_len = len(curr_sequence)

            for curr_token in curr_sequence:
                self.word_freq[curr_token] = self.word_freq.get(curr_token, 0) + 1

        initial_special_tokens = {self.pad_token, self.start_token, self.end_token, self.unk_token}
        for curr_token, curr_freq in self.word_freq.items():
            if curr_freq >= self.freq_threshold and curr_token not in initial_special_tokens:
                self.add_word(curr_token)
        print(f'Vocabulary built with {len(self)} words.')

    def __call__(self, word):
        '''Looks up a word's index, returns <UNK> index if not found.'''
        return self.word2idx.get(word, self.unk_idx)

    def __len__(self):
        '''Returns the total size of the vocabulary.'''
        return len(self.word2idx)


def preprocess_caption_for_decoder(raw_caption, vocab):
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

    if not raw_caption:
        # Handle empty caption
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, 0,
                                                                                                  dtype=torch.long)

    # Tokenize and Calculate Original Word Counts
    tokens = nltk.word_tokenize(raw_caption.lower())
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
    max_len_input_target = vocab.max_caption_len + 1

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


if __name__ == '__main__':
    # Preprocess captions
    # Make a list of all train captions
    train_captions = []
    with open(csv_train_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            train_captions.extend(row[1:6])
    # Tokenize train captions
    tokenized_captions = [nltk.word_tokenize(s.lower()) for s in train_captions]

    # Make vocabulary out of train captions
    miniCOCO_vocabulary = Vocabulary(freq_threshold=1)
    miniCOCO_vocabulary.build_vocabulary(sentences_list=tokenized_captions)

