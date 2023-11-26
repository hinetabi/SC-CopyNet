import torch
from collections import Counter
from torchtext.vocab import vocab, Vocab
import io
import argparse
import logging
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-file', action='store', dest='source_file', 
                    help='Path to source data for build vocab')
    
    parser.add_argument('--save-file', action='store', dest='save_file',
                    help='Path to save_file for saving the vocab')
    
    parser.add_argument('--min-vocab', action='store', dest='min_vocab', default=1,
                    help='Min frequency of each vocab in vocab list.')

    
    opt = parser.parse_args()
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    logging.info(opt)


    vi_vocab = VietVocab(source_file=opt.source_file)
    vi_vocab.build_vocab(min_freq=opt.min_vocab)
    vi_vocab.save_vocab(save_file=opt.save_file)