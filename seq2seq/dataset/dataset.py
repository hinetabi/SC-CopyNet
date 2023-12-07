import torch
import io
from torch.utils.data import Dataset
from seq2seq.util.utils import tokenize
from torchtext.vocab import Vocab

class ViDataset(Dataset):
    """Generate dataset from source path, target path and vocab (torchtext)
    """
    def __init__(self, vocab: Vocab, src_path: str, tgt_path: str) -> None:
        self.vocab = vocab
        
        with open(src_path, encoding="utf-8") as f:
            src = f.readlines()

        with open(tgt_path, encoding="utf-8") as f:
            tgt = f.readlines()

        # Combine src and tgt into pairs
        combined_data = list(zip(src, tgt))

        # Sort based on the length of src (you can use len(src) or len(tgt) depending on your preference)
        sorted_data = sorted(combined_data, key=lambda x: len(tokenize(x[0])), reverse=True)

        # Unpack the sorted data back into separate src and tgt lists
        self.raw_src_iter, self.raw_tgt_iter = zip(*sorted_data)
            
        assert len(self.raw_src_iter) == len(self.raw_tgt_iter), "Source and Target file should have the same length."
        
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

        return source_indices, target_indices