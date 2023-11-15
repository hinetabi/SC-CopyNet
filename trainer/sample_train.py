import os
import argparse
import logging
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.vocab import VietVocab
from dataset.dataset import ViDataset
from model import GRUEncoder, GRUDecoder, Attention, Seq2Seq
from util.utils import generate_batch, init_weights, count_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--src-train-path', action='store', dest='src_train_path',required=True,
                    help='Path to source train data')
parser.add_argument('--tgt-train-path', action='store', dest='tgt_train_path',required=True,
                    help='Path to target train data')
parser.add_argument('--src-test-path', action='store', dest='src_test_path',required=False,
                    help='Path to source test data')
parser.add_argument('--tgt-test-path', action='store', dest='tgt_test_path',required=False,
                    help='Path to target test data')

parser.add_argument('--vocab-path', action='store', dest='vocab_path',required=True,
                    help='Path to vocab file (.pth)')

parser.add_argument('--save-model-dir', action='store', dest='save_model_dir', default='./models',
                    help='Path to save best model.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logging.info(opt)

# get vocab
viet_vocab = VietVocab()
viet_vocab.load_vocab(save_file=opt.vocab_path)
vi_vocab = viet_vocab.get_vocab()

# set const params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 4
PAD_IDX = vi_vocab['<pad>']
BOS_IDX = vi_vocab['<bos>']
EOS_IDX = vi_vocab['<eos>']

# create dataset
train_data = ViDataset(vocab=vi_vocab, src_path=opt.src_train_path, tgt_path=opt.tgt_train_path)
test_data = ViDataset(vocab=vi_vocab, src_path=opt.src_test_path, tgt_path=opt.tgt_test_path)
val_data = ViDataset(vocab=vi_vocab, src_path=opt.src_test_path, tgt_path=opt.tgt_test_path)


# create dataloader
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=lambda x: generate_batch(x, BOS_IDX, EOS_IDX, PAD_IDX, device))
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=lambda x: generate_batch(x, BOS_IDX, EOS_IDX, PAD_IDX, device))
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=lambda x: generate_batch(x, BOS_IDX, EOS_IDX, PAD_IDX, device))

# create models
INPUT_DIM = len(vi_vocab.vocab)
OUTPUT_DIM = len(vi_vocab.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

# model preparation
model.apply(init_weights)
print(f'The model has {count_parameters(model):,} trainable parameters')

# create optimizer
optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = vi_vocab['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# function train
def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch[0]
        trg = batch[1]

        optimizer.zero_grad()

        output = model(src, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch[0]
            trg = batch[1]

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# epoch time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# from torchtext.data import BucketIterator
BATCH_SIZE = 4

# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, val_data, test_data),
#     batch_size = BATCH_SIZE,
#     device = device)

train_iterator, valid_iterator, test_iterator = train_iter, valid_iter, test_iter

N_EPOCHS = 2
CLIP = 1

import time
import math

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')