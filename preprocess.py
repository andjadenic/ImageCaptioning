import nltk
import torch
import string
from config import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image


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
        - input_tensor (torch.Tensor): [id(<START>), id(w1), ..., id(wN)]
            Indexed and padded input sequence.
            Includes <START> index, excludes <END> index.
        - length_tensor (torch.Tensor): Original sequence lengths + 1.
            Required by pack_padded_sequence for DecoderRNN's forward method.
            size = ([batch_size, ])
        - target_tensor (torch.Tensor):  [id(w1), ..., id(wN), id(<END>)]
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
    input_tensor = torch.tensor(padded_input, dtype=torch.long).to(device)
    target_tensor = torch.tensor(padded_target, dtype=torch.long).to(device)
    length_tensor = torch.tensor(len(tokens) + 1, dtype=torch.long).to(device)

    return {'input_tensor': input_tensor,
            'length_tensor': length_tensor,
            'target_tensor': target_tensor}


def collate_fn(input_batch):
    '''Convert list of getitem's returns into mini-batch torch tensors in descending order by length.'''
    input_batch.sort(key=lambda x: x['length_tensor'].item(), reverse=True)

    img_tensor = [item['img_tensor'] for item in input_batch]
    input_tensor = [item['input_tensor'] for item in input_batch]
    length_tensor = [item['length_tensor'] for item in input_batch]
    target_tensor = [item['target_tensor'] for item in input_batch]

    return {
        'img_tensor': torch.stack(img_tensor).to(torch.float),
        'input_tensor': torch.stack(input_tensor).to(torch.long),
        'length_tensor': torch.stack(length_tensor).to(torch.long),
        'target_tensor': torch.stack(target_tensor).to(torch.long)
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

        with open(train_annFile, 'r') as file:
            data = json.load(file)
        self.captions_id_list = [item['id'] for item in data['annotations']]

    def __len__(self):
        return len(self.captions_id_list)

    def __getitem__(self, idx):
        sample = get_ith_sample(id=idx, annFile=self.json_path, data_path=self.data_path)

        raw_img = sample['img']
        raw_caption = sample['caption']

        preprocessed_caption = preprocess_caption(raw_caption, self.vocabulary)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transformed_img = transform(to_pil_image(raw_img))

        return {
            'img_tensor': transformed_img.to(device),
            'input_tensor': preprocessed_caption['input_tensor'],
            'length_tensor': preprocessed_caption['length_tensor'],
            'target_tensor': preprocessed_caption['target_tensor']
        }



if __name__ == '__main__':
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
        collate_fn=collate_fn
    )

    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        print(f'batch {i}:')

        '''packed_input = torch.nn.utils.rnn.pack_padded_sequence(
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