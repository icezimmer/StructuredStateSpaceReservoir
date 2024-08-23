# import torch
# from transformers import BertTokenizer
# from torch.utils.data import Dataset


# class TextDataset(Dataset):
#     def __init__(self, dataset, max_length):
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#         self.data = []
#         for label, text in dataset:
#             print("text", text)
#             self.data.append(self.preprocess_function(text, label, max_length))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]
    
#     def preprocess_function(self, text, label, max_length):
#         encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
#         return encoding['input_ids'].squeeze(0), torch.tensor(0 if label == 'neg' else 1)


import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


class TextDataset(Dataset):
    def __init__(self, dataset, max_length, level, min_freq, append_bos, append_eos):
        self.level = level
        self.max_length = max_length
        self.append_bos = append_bos
        self.append_eos = append_eos

        # Choose tokenizer based on the level
        self.tokenizer = list if level == 'char' else get_tokenizer('basic_english')
        
        # Step 1: Build tokens list and vocab simultaneously
        tokens_list = []
        self.data = []
        
        for label, text in dataset:
            tokens = self.tokenizer(text)
            tokens_list.append(tokens)  # Collect tokens for vocabulary building
            self.data.append((label, tokens))  # Store the label and tokenized text

        # Step 2: Build the vocabulary
        self.vocab = build_vocab_from_iterator(tokens_list, min_freq=min_freq, specials=['<pad>', '<unk>', '<bos>', '<eos>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        # Step 3: Preprocess the dataset
        self.data = [self.preprocess_function(tokens, label) for label, tokens in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess_function(self, tokens, label):
        # Optionally add special tokens
        if self.append_bos:
            tokens = ['<bos>'] + tokens
        if self.append_eos:
            tokens = tokens + ['<eos>']
        
        # Numericalize tokens
        token_ids = self.vocab(tokens)

        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.vocab['<pad>']] * (self.max_length - len(token_ids))
        
        # Convert to tensor and return with label
        return torch.tensor(token_ids), torch.tensor(0 if label == 'neg' else 1)


