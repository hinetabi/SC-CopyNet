import torch
import io
from torch.utils.data import Dataset
from util.utils import tokenize
from torchtext.vocab import Vocab

class CustomDataset(Dataset):
    """Generate dataset from source path, target path and vocab (torchtext)
    """
    def __init__(self, vocab: Vocab, src_path: str, tgt_path: str) -> None:
        self.vocab = vocab
        
        with open(src_path, encoding="utf-8") as f:
            self.raw_src_iter = f.readlines()

        with open(tgt_path, encoding="utf-8") as f:
            self.raw_tgt_iter = f.readlines()
            
        assert len(self.raw_src_iter) == len(self.raw_tgt_iter), "Source and Target file should have the same length."
        print(self.raw_src_iter)

    def __len__(self):
        return len(list(self.raw_src_iter))

    def __getitem__(self, index):
        source_sentence = self.raw_src_iter[index]
        target_sentence = self.raw_tgt_iter[index]

        # tokenize source_sentence & target_sentence
        source_tokenize = tokenize(source_sentence)
        target_tokenize = tokenize(target_sentence)

        # Convert sentences to indices using vocabulary
        source_indices = [self.vocab[word] for word in source_tokenize]
        target_indices = [self.vocab[word] for word in target_tokenize]

        return torch.tensor(source_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)