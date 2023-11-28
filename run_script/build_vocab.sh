#! /bin/sh

SOURCE_FILE=/home/hinetabi/Documents/university/data/sample/data/train_tgt.txt
SAVE_FILE=tests/data/vocab.pth
MIN_VOCAB=1
export PYTHONPATH=./

# build the vocab
python run/build_vocab.py --source-file $SOURCE_FILE --save-file $SAVE_FILE --min-vocab $MIN_VOCAB