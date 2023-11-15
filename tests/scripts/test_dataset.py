from dataset.dataset import ViDataset
from dataset.vocab import VietVocab

vi_vocab = VietVocab()
vi_vocab.load_vocab(save_file="tests/data/vocab.pth")

train_dataset = ViDataset(vocab=vi_vocab.get_vocab(), src_path="tests/data/src.txt", tgt_path="tests/data/tgt.txt")

len1 = train_dataset.__len__()

for i in range(len1):
    print(train_dataset[i])