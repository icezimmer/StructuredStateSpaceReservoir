import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, dataset, max_length):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.data = []
        for label, text in dataset:
            self.data.append(self.preprocess_function(text, label, max_length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess_function(self, text, label, max_length):
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        return encoding['input_ids'].squeeze(0), torch.tensor(0 if label == 'neg' else 1)
