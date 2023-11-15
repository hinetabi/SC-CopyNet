import torch
from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


BATCH_SIZE = 128

def generate_batch(data_batch, BOS_IDX, EOS_IDX, PAD_IDX, device):
    de_batch, en_batch = [], []
    for data in data_batch:
        de_item, en_item = data[0], data[1]
        print(de_item, en_item)
        de_batch.append(torch.cat([torch.tensor([BOS_IDX], device=device), de_item.to(device), torch.tensor([EOS_IDX], device=device)], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX], device=device), en_item.to(device), torch.tensor([EOS_IDX], device=device)], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch

def tokenize(sentence: str) -> List[str]:
    """Tokenize the sentence by space and punctuation

    Args:
        sentence (str): input sentence

    Returns:
        List[str]: list of tokenized string.
    """

    return sentence.split()

def init_weights(m):
    """Initilize weights before training
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    """Count parameters from model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)