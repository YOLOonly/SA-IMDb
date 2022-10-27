import os
import torch
import random
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms as T
from torch.utils.data import TensorDataset


def read_imdb(path='./aclImdb', is_train=True):
    reviews, labels = [], []
    tokenizer = get_tokenizer('basic_english')
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, 'train' if is_train else 'test', label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name, filename), mode='r', encoding='utf-8') as f:
                reviews.append(tokenizer(f.read()))
                labels.append(1 if label == 'pos' else 0)
    return reviews, labels


def build_dataset(reviews, labels, max_len=512):
    vocab = build_vocab_from_iterator(reviews, min_freq=2, specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.Truncate(max_seq_len=max_len),
        T.ToTensor(padding_value=vocab['<pad>']),
        T.PadTransform(max_length=max_len, pad_value=vocab['<pad>']),
    )
    dataset = TensorDataset(text_transform(reviews), torch.tensor(labels))
    return dataset, vocab


reviews, labels = read_imdb()
print(sum(list(map(lambda x: len(x), reviews))) / len(reviews))
