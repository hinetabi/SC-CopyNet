#! /bin/sh

SOURCE_TRAIN_FILE=tests/data/src.txt
TARGET_TRAIN_FILE=tests/data/tgt.txt
VOCAB_FILE=tests/data/vocab.pth

export PYTHONPATH=./

# build the vocab
python seq2seq/trainer/sample_train.py --src-train-path $SOURCE_TRAIN_FILE --tgt-train-path $TARGET_TRAIN_FILE --src-test-path $SOURCE_TRAIN_FILE --tgt-test-path $TARGET_TRAIN_FILE --vocab-path $VOCAB_FILE