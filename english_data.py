from torchtext.data import get_tokenizer
import torchtext.vocab as vocab
from torch.utils.data import Dataset
from typing import Union, Iterable
import torch
from torch.nn.utils.rnn import pad_sequence

UNKNOWN_TOKEN = "<{UNKNOWN_TOKEN}>"
START_TOKEN = "<{START_TOKEN}>"
END_TOKEN = "<{END_TOKEN}>"
SEP_TOKEN = "<{SEP_TOKEN}>"
SPECIALS = [UNKNOWN_TOKEN, START_TOKEN, END_TOKEN]


class EnglishData(Dataset):
    def __init__(self, english_data_iterator: Union[Iterable[str], Iterable[Iterable[str]]],
                 minimum_frequency: Union[int, None] = None,
                 vocabulary: Union[vocab.vocab, None] = None, is_tokenized: bool = False):
        if minimum_frequency is None and vocabulary is None:
            raise TypeError("Need either minimum frequency to build vocabulary or a vocabulary.")
        self.tokenizer = get_tokenizer("basic_english")
        self.data: list[list[str]] = [
            [token.lower().strip() for token in self.tokenizer(token_set)] if not is_tokenized else list[token_set]
            for token_set in english_data_iterator
        ]

        if vocabulary is None:
            self.vocabulary = vocab.build_vocab_from_iterator(self.data, specials=SPECIALS)
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary

        self.encoded_data = [[self.vocabulary[token] for token in token_set] for token_set in self.data]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.encoded_data[index][:-1]).long(), torch.tensor(self.encoded_data[index][1:]).long()

    def collate(self, data: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        x_list, y_list = [], []
        for x, y in data:
            x_list.append(x)
            y_list.append(y)

        return pad_sequence(x_list, batch_first=True, padding_value=self.vocabulary[END_TOKEN]), \
            pad_sequence(y_list, batch_first=True, padding_value=self.vocabulary[END_TOKEN])
