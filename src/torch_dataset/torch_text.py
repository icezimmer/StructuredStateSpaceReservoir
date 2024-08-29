import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


class TextDataset(Dataset):
    def __init__(self, dataset, max_length, level, min_freq, append_bos, append_eos, padding_idx):
        self.level = level
        self.max_length = max_length
        self.append_bos = append_bos
        self.append_eos = append_eos
        self.padding_idx = padding_idx

        # Choose tokenizer based on the level
        self.tokenizer = list if level == 'char' else get_tokenizer('basic_english')
        
        # Step 1: Build tokens list and vocab simultaneously
        tokens_list = []
        self.data = []
        
        for label, text in dataset:
            tokens = self.tokenizer(text)
            tokens_list.append(tokens)  # Collect tokens for vocabulary building
            self.data.append((tokens, label))  # Store the label and tokenized text

        # Step 2: Build the vocabulary
        self.vocab = build_vocab_from_iterator(tokens_list, min_freq=min_freq, specials=['<pad>', '<unk>', '<bos>', '<eos>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        # Step 3: Preprocess the dataset
        self.data = [self._preprocess_function(tokens, label) for tokens, label in self.data]

    def _compute_lengths(self, inputs):
        """
        Compute the length of each padded time series in a batch.

        :param inputs: Tensor of shape (L,), where L is the sequence length.
        :return: Tensor of shape (), containing the lengths of the sequence.
        """
        # Create a mask where non-padding elements are True
        non_padding_mask = inputs != self.padding_idx  # Shape: (L)

        # Sum along the time dimension to get the length of each sequence
        lengths = non_padding_mask.sum(dim=-1)  # Shape: (,)

        return lengths

    def _preprocess_function(self, tokens, label):
        # Numericalize tokens
        token_ids = self.vocab(tokens)
        new_length = self.max_length - int(self.append_bos) - int(self.append_eos)

        # Truncate or pad to max_length
        if len(token_ids) > new_length:
            token_ids = token_ids[:new_length]

        # Append <bos> and <eos> if required
        if self.append_bos:
            token_ids = [self.vocab['<bos>']] + token_ids
        if self.append_eos:
            token_ids = token_ids + [self.vocab['<eos>']]

        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.vocab['<pad>']] * (self.max_length - len(token_ids))

        tokens = torch.tensor(token_ids)
        label = torch.tensor(0 if label == 'neg' else 1)
        lengths = self._compute_lengths(tokens)

        return tokens, label, lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
