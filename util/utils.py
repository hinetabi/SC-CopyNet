import torch
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence


BATCH_SIZE = 128

def generate_batch(data_batch, BOS_IDX, EOS_IDX, PAD_IDX):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
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