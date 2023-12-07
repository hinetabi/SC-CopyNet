import os
import argparse
import logging
import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchtext
import bitsandbytes as bnb

# set path
import sys
sys.path.append('./')

from seq2seq.optim import Optimizer
from seq2seq.trainer import SupervisedTrainer
from seq2seq.model import GRUDecoder, GRUEncoder, Attention, Seq2Seq
from seq2seq.loss import Perplexity, CrossEntropyLoss
from seq2seq.dataset.dataset import ViDataset
from seq2seq.dataset.vocab import VietVocab
from seq2seq.evaluator.predictor import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.utils import generate_batch, init_weights, count_parameters

raw_input = input  # Python 3

# Sample usage:
#     # training
#     python run/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python run/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python run/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--src-train-path', action='store', dest='src_train_path',required=False,
                    help='Path to source train data')
parser.add_argument('--tgt-train-path', action='store', dest='tgt_train_path',required=False,
                    help='Path to target train data')
parser.add_argument('--src-val-path', action='store', dest='src_val_path',required=False,
                    help='Path to source test data')
parser.add_argument('--tgt-val-path', action='store', dest='tgt_val_path',required=False,
                    help='Path to target test data')

parser.add_argument('--vocab-path', action='store', dest='vocab_path',required=False,
                    help='Path to vocab file (.pth)')

parser.add_argument('--expt-dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to save best model.')

parser.add_argument('--batch-size', action='store', dest='batch_size', default=32,
                    help='Batch_size.')

parser.add_argument('--num-epochs', action='store', dest='num_epochs', default=5,
                    help='Number of training epochs')

parser.add_argument('--load-checkpoint', action='store', dest='load_checkpoint',required=False, default=None,
                    help='Path to load last checkpoint.')

parser.add_argument('--checkpoint-every', action='store', dest='checkpoint_every',required=False, default=1000,
                    help='Path to load last checkpoint.')

parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()
# opt.batch_size = int(opt.batch_size)
# opt.num_epochs = int(opt.num_epochs)
# opt.checkpoint_every = int(opt.checkpoint_every)

import os

opt.batch_size = 2
opt.num_epochs = 100
opt.src_train_path = "/home/hinetabi/Documents/university/data/sample/data/train_src.txt"
opt.tgt_train_path = "/home/hinetabi/Documents/university/data/sample/data/train_tgt.txt"
opt.vocab_path = "tests/data/vocab.pth"
opt.src_val_path = "/home/hinetabi/Documents/university/data/sample/data/val_src.txt"
opt.tgt_val_path = "/home/hinetabi/Documents/university/data/sample/data/val_tgt.txt"



LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    
    predictor = Predictor(model, input_vocab, output_vocab)

    while True:
        seq_str = raw_input("Type in a source sequence:")
        seq = seq_str.strip().split()
        print(predictor.predict(seq))
else:
    # get vocab
    viet_vocab = VietVocab()
    viet_vocab.load_vocab(save_file=opt.vocab_path)
    vi_vocab = viet_vocab.get_vocab()

    # set const params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PAD_IDX = vi_vocab['<pad>']
    BOS_IDX = vi_vocab['<bos>']
    EOS_IDX = vi_vocab['<eos>']
    TRG_PAD_IDX = vi_vocab['<pad>']
    
    # create dataset & dataloader
    train_data = ViDataset(vocab=vi_vocab, src_path=opt.src_train_path, tgt_path=opt.tgt_train_path)
    train_iter = DataLoader(train_data, batch_size=opt.batch_size,
                            shuffle=True, collate_fn=lambda x: generate_batch(x, BOS_IDX, EOS_IDX, PAD_IDX, device))
    
    val_iter = None
    if opt.src_val_path is not None:
        val_data = ViDataset(vocab=vi_vocab, src_path=opt.src_val_path, tgt_path=opt.tgt_val_path)
        val_iter = DataLoader(val_data, batch_size=opt.batch_size,
                        shuffle=True, collate_fn=lambda x: generate_batch(x, BOS_IDX, EOS_IDX, PAD_IDX, device))
    
    logging.info(f"train_iter = {train_iter}")
    logging.info(f"val_iter = {val_iter}")
    
    # set const value
    INPUT_DIM = len(vi_vocab.vocab)
    OUTPUT_DIM = len(vi_vocab.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # loss function
    # torch.nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    weight = torch.ones(len(vi_vocab.vocab), device=device)
    loss = CrossEntropyLoss(weight, TRG_PAD_IDX=TRG_PAD_IDX)
    loss.to(device)

    model = None
    optimizer = None
    # Initialize model
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)
    # model preparation
    model.apply(init_weights)
    logging.info(f'The model has {count_parameters(model):,} trainable parameters')

    # create optimizer
    optimizer = Optimizer(bnb.optim.AdamW(model.parameters()), max_grad_norm=0)
    scheduler = StepLR(optimizer.optimizer, 1)
    optimizer.set_scheduler(scheduler)
    
    # train
    t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                          checkpoint_every=opt.checkpoint_every,
                          print_every=100, expt_dir=opt.expt_dir)
    
    # Initialize WandB
    wandb.init(project='spelling-error-correction', config={})
    model = t.train(model, train_iter,
                      vi_vocab=vi_vocab,
                      num_epochs=opt.num_epochs, 
                      val_iter=val_iter,
                      optimizer=optimizer)
    wandb.finish()