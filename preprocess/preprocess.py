import nltk
import string
from utils.config import *
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import os
import time
from torch.nn.utils.rnn import pad_sequence


def make_dataset(ann_json, data_path, new_json_path):
    '''
    Function reads information about images and captions
    from ann_json and data_path
    and saves the dataset in new_json_path
    where a single sample is an image and 5 corresponding captions
    '''
    start_time = time.time()

    with open(ann_json, 'r') as file:
        data = json.load(file)

    dataset = []
    for img_info in data['images']:
        curr_img_name = img_info['file_name']
        curr_img_id = img_info['id']
        curr_img_path = os.path.join(data_path, curr_img_name)
        curr_captions_list = []
        for ann in data['annotations']:
            if ann['image_id'] == curr_img_id:
                curr_captions_list.append(ann['caption'])

        curr_sample = {
            'img_id': curr_img_id,
            'img_name': curr_img_name,
            'img_path': curr_img_path,
            'captions': curr_captions_list
        }
        dataset.append(curr_sample)

    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f'Evaluation Dataset is successfully created in {time_elapsed:.2f} secunds and saved in {new_json_path}.')

    return dataset


def make_dataset_1c(dataset_path, dataset_1c_path):
    '''
    Function reads information about images and captions
    from dataset_path
    and saves the dataset_1c in dataset_1c_path
    where a single sample is an image and one corresponding caption
    '''
    dataset = read_dataset(dataset_path)
    dataset_1c = []
    for sample in dataset:
        curr_img_name = sample['img_name']
        curr_img_id = sample['img_id']
        curr_img_path = sample['img_path']
        for curr_caption in sample['captions']:
            dataset_1c.append({
                'img_id': curr_img_id,
                'img_name': curr_img_name,
                'img_path': curr_img_path,
                'caption': curr_caption
            })
    open(dataset_1c_path, "w", encoding="utf-8").write(json.dumps(dataset_1c))
    print(f'Dataset 1C successfully created in {dataset_1c_path}.')
    return dataset_1c


def read_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx = 0

        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        self.pad_idx = self.add_word(self.pad_token)  # id(<PAD>) = 0
        self.start_idx = self.add_word(self.start_token)  # id(<START>) = 1
        self.end_idx = self.add_word(self.end_token)  # id(<END>) = 2
        self.unk_idx = self.add_word(self.unk_token)  # id(<UNK>) = 3

    def add_word(self, word):
        '''Adds a word to the vocabulary
        :return: index of added word'''
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word.append(word)
            self.idx += 1
            return self.idx - 1  # Return the index assigned
        return self.word2idx[word]  # Return existing index

    def build_vocabulary(self, captions_path, vocab_path):
        '''Builds the vocabulary from provided captions_path annotations
        and saves word2id and id2word mappings in vocab_json'''
        start_time = time.time()

        print('Building vocabulary...')

        print(f"Loading annotations from {captions_path}...")
        with open(captions_path, 'r', encoding='utf-8') as f:
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
            tokenized_caption = caption.split()
            tokenized_captions.append(tokenized_caption)

        # Add all words to the vocabulary
        for caption in tokenized_captions:
            for token in caption:
                self.add_word(token)

        # Save word2id and id2word mappings
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump([self.word2idx, self.idx2word], f, indent=4, ensure_ascii=False)

        end_time = time.time()

        print(f'Vocabulary built in {(end_time - start_time):.2f} seconds with {len(self)} words and saved in {vocab_path}.')

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            word2id, id2word = json.load(f)
        self.word2idx = word2id
        self.idx2word = id2word
        print('Vocabulary successfully loaded.')

    def __call__(self, word):
        '''Looks up a word's index, returns <UNK> index if not found.'''
        return self.word2idx.get(word, self.unk_idx)

    def __len__(self):
        '''Returns the total size of the vocabulary.'''
        return len(self.word2idx)


def preprocess_caption(raw_caption, vocab):
    '''
    Preprocesses a single raw caption for DecoderRNN.
    Pipeline: removes punctuation, converts words to lowercase, index words,
    adds <START> and <END> tokens

    :param raw_captions (str): Raw caption
    :param vocab: A Vocabulary object
    '''

    # Remove punctuation, lower case.
    translator = str.maketrans('', '', string.punctuation)
    processed_caption = raw_caption.lower().translate(translator)

    tokens = processed_caption.lower().split()

    # Numericalize, Add Special Tokens (<START>, <END>)
    caption_indices = [vocab(token) for token in tokens if vocab(token) != vocab.unk_idx]  # Maps tokens into indexes: [id(w1), id(w2), ..., id(wN)]
    full_indices = [vocab.start_idx] + caption_indices + [vocab.end_idx]   # Includes <START> and <END> indexes:
                                                         # [id(<START>), id(w1), id(w2), ..., id(wN), id(<END>)]

    # Convert to PyTorch Tensor
    output_tensor = torch.tensor(full_indices, dtype=torch.long)

    return output_tensor


class CocoDataset(Dataset):
    '''
    CHANGE
    In this dataset single sample is an image and its 5 corresponding caption
    '''
    def __init__(self, data_path, dataset_path, vocabulary):
        self.data_path = data_path
        self.dataset_path = dataset_path

        self.vocabulary = vocabulary
        self.word2idx = vocabulary.word2idx
        self.idx2word = vocabulary.idx2word
        self.vocab_size = len(vocabulary)

        self.dataset = read_dataset(self.data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        sample = self.dataset[id]

        img_path = sample['img_path']
        raw_captions = sample['captions']

        # Preprocess the image
        raw_img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(raw_img)

        # Preprocess raw captions
        #preprocessed_caption = preprocess_caption(raw_caption, self.vocabulary)
        # DODAJ KAKO DA SE SJEDINE INDEKSIRANI OPISI

        return


transform_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class CocoDataset1c(Dataset):
    '''
    In this dataset single sample is an image and a single corresponding captions
    '''
    def __init__(self, data_path, dataset_1c_path, vocabulary, transform=None, target_transform=None):
        self.data_path = data_path
        self.dataset_1c_path = dataset_1c_path

        self.vocabulary = vocabulary
        self.word2idx = vocabulary.word2idx
        self.idx2word = vocabulary.idx2word
        self.vocab_size = len(vocabulary)

        self.dataset_1c = read_dataset(self.dataset_1c_path)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset_1c)

    def __getitem__(self, id):
        sample = self.dataset_1c[id]

        img_path = sample['img_path']
        raw_caption = sample['caption']

        # Preprocess the image
        raw_img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(raw_img)

        # Preprocess raw caption
        if self.target_transform is not None:
            caption = self.target_transform(raw_caption, self.vocabulary)
        length = caption.shape[0] - 2

        return {
            'img_tensor': img,
            'cap_tensor': caption,
            'length_tensor': torch.tensor(length).to(torch.long)
        }


def collate_fn_1d(samples_list):
    samples_list.sort(key=lambda x: x['length_tensor'].item(), reverse=True)

    img_tensor = [item['img_tensor'] for item in samples_list]
    cap_tensor = [item['cap_tensor'] for item in samples_list]
    length_tensor = [item['length_tensor'] for item in samples_list]

    return {
        'img_tensor': torch.stack(img_tensor).to(torch.float),
        'cap_tensor': pad_sequence(cap_tensor, batch_first=True).to(torch.long),
        'length_tensor': torch.stack(length_tensor).to(torch.long)
    }


if __name__ == '__main__':
    # Make JSON dataset files for training and validation data (Run once)
    # train_dataset's sample is an image with 5 corresponding captions
    # train_dataset_1c's sample is an image with a single corresponding captions
    '''train_dataset = make_dataset(ann_json=captions_train_path,
                                    data_path=train_data_path,
                                    new_json_path=train_dataset_json)
        make_dataset_1c = make_dataset_1c(dataset_path=train_dataset_json,
                                      dataset_1c_path=train_dataset_1c_json)
    val_dataset = make_dataset(ann_json=captions_val_path,
                               data_path=val_data_path,
                               new_json_path=val_dataset_json)
    val_dataset_1c = make_dataset_1c(dataset_path=val_dataset_json,
                                     dataset_1c_path=val_dataset_1c_json)'''

    # Make vocabulary
    vocabulary = Vocabulary()
    # Build vocabulary out of training data (run only once)
    '''nltk.download('punkt_tab')
    vocabulary.build_vocabulary(captions_path=captions_train_path,
                                vocab_path=vocab_json)'''

