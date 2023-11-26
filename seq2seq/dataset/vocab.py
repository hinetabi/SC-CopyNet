import torch
from collections import Counter
from torchtext.vocab import vocab, Vocab
import io
from seq2seq.util.utils import tokenize

class VietVocab(object):
    """
    Build the vocab from txt file.
    Each row of txt file is one sentence.
    """
    def __init__(self, source_file: str = None) -> None:
        self.source_file = source_file
        self.vi_vocab = None

    def build_vocab(self, min_freq: int):
        """
        Build the vocab from filepath
        """
        counter = Counter()
        with io.open(self.source_file, encoding="utf-8") as f:
            for string_ in f:
                counter.update(tokenize(string_))
        self.vi_vocab = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq = min_freq)
        self.vi_vocab.set_default_index(self.vi_vocab['<unk>'])

    def save_vocab(self, save_file):
        torch.save(self.vi_vocab, save_file)

    def load_vocab(self, save_file: str):
        self.vi_vocab = torch.load(save_file)
    
    def get_vocab(self) -> Vocab:
        return self.vi_vocab