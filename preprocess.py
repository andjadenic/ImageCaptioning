import nltk
import torch
import string
from config import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *


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
    Preprocesses single raw caption for DecoderRNN.
    Pipeline: removes punctuation, converts words to lowercase, numericalizes words,
    adds <START> and <END> tokens, adds padding

    :param raw_captions (str): Raw caption
    :param vocab: A Vocabulary object

    :return:
    dictionary: A dictionary containing:
        - inputs_tensor (torch.Tensor): [id(<START>), id(w1), ..., id(wN)]
            Indexed and padded input sequence.
            Includes <START> index, excludes <END> index.
        - lengths_tensor (torch.Tensor): Original sequence lengths + 1.
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
    inputs_tensor = torch.tensor(padded_input, dtype=torch.long)
    targets_tensor = torch.tensor(padded_target, dtype=torch.long)
    lengths_tensor = torch.tensor(len(tokens) + 1, dtype=torch.long)

    return {'inputs_tensor': inputs_tensor,
            'lengths_tensor': lengths_tensor,
            'targets_tensor': targets_tensor}


def collate_fn(input_batch, vocabulary):
    '''Convert list of preprocessed images and captions into mini-batch torch tensors using pre-built vocabulary in descending order.'''
    input_batch.sort(key=lambda x: x['lengths_tensor'], reverse=True)

    img_tensor = [item['img_tensor'] for item in input_batch]
    inputs_tensor = [item['inputs_tensor'] for item in input_batch]
    lengths_tensor = [item['lengths_tensor'] for item in input_batch]
    targets_tensor = [item['targets_tensor'] for item in input_batch]

    return {
        'img_tensor': torch.stack(img_tensor),
        'inputs_tensor': torch.stack(inputs_tensor).to(torch.long),
        'lengths_tensor': torch.stack(lengths_tensor).to(torch.long),
        'targets_tensor': torch.stack(targets_tensor).to(torch.long)
    }


class CocoDataset(Dataset):
    def __init__(self, data_path, json_path, vocabulary):
        self.data_path = data_path
        self.json_path = json_path

        self.vocabulary = vocabulary
        self.word2idx = vocabulary.word2idx
        self.idx2word = vocabulary.idx2word
        self.L = vocabulary.L
        self.vocab_size = len(vocabulary)

        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        image_names_sorted = [f for f in os.listdir(self.data_path)
                       if f.lower().endswith(supported_extensions)]
        image_names_sorted.sort()
        img_paths = [os.path.join(self.data_path, name) for name in image_names_sorted]
        images_sorted = [io.imread(img_path) for img_path in img_paths]
        self.img_list = []
        self.captions_list = []
        for img, img_name in zip(images_sorted, img_paths):
            captions = get_captions(self.json_path, img_name)
            for caption in captions:
                self.img_list.append(img)
                self.captions_list.append(caption)

    def __len__(self):
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        image_files = [f for f in os.listdir(self.data_path)
                       if f.lower().endswith(supported_extensions)]
        return len(image_files)

    def __getitem__(self, idx):
        img_name = get_ith_image(self.data_path, idx)['img_name']
        img = get_ith_image(self.data_path, idx)['img']
        captions_list = get_captions(self.json_path, img_name)

        preprocessed_captions = [preprocess_caption(c, self.vocabulary) for c in captions_list]
        inputs_tensor = [item['inputs_tensor'] for item in preprocessed_captions]
        lengths_tensor = [item['lengths_tensor'] for item in preprocessed_captions]
        targets_tensor = [item['targets_tensor'] for item in preprocessed_captions]

        return {
            'img_tensor': torch.tensor(img).to(torch.long),
            'inputs_tensor': torch.stack(inputs_tensor).to(torch.long),
            'lengths_tensor':  torch.stack(lengths_tensor).to(torch.long),
            'targets_tensor': torch.stack(targets_tensor).to(torch.long)
        }



if __name__ == '__main__':
    vocabulary = Vocabulary()
    vocabulary.build_vocabulary(json_path=train_annFile)

    train_dataset = CocoDataset(data_path=train_data_path,
                                json_path=train_annFile,
                                vocabulary=vocabulary)

    print(f'{len(train_dataset.img_list)=}')
    print(f'{len(train_dataset.captions_list)=}')
    #print(train_dataset[0]['img_tensor'])

    '''train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    print(train_loader.batch_size)
    for i, batch in enumerate(train_loader):
        if i >=3:
            break
        print(f'batch {i}:')
        print(f'{batch['inputs_tensor']=}')
        print(f'{batch['lengths_tensor']=}')

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            input=batch['processed_sentence'],
            lengths=batch['length'],
            batch_first=True
        )
        print(f'{packed_input.data=}')
        print(f'{packed_input.batch_sizes=}')

        unpacked_output, unpacked_length = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=packed_input,
            batch_first=True
        )
        print(f'{unpacked_output=}', '\n')'''