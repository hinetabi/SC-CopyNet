#! /bin/sh

SOURCE_FILE=tests/data/tgt.txt
SAVE_FILE=tests/data/vocab.pth
export PYTHONPATH=./

# build the vocab
python run/build_vocab.py --source-file $SOURCE_FILE --save-file $SAVE_FILE