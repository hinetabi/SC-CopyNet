#! /bin/sh

SOURCE_TRAIN_FILE=tests/data/src.txt
TARGET_TRAIN_FILE=tests/data/tgt.txt
VOCAB_FILE=tests/data/vocab.pth
CHECKPOINT_DIR=/home/hinetabi/Documents/university/copynet/SC-CopyNet/experiment/checkpoints
EXPT_PATH=/home/hinetabi/Documents/university/copynet/SC-CopyNet/experiment

export PYTHONPATH=./

# build the vocab
python run/main.py --src-train-path $SOURCE_TRAIN_FILE --tgt-train-path $TARGET_TRAIN_FILE --vocab-path $VOCAB_FILE --src-val-path $SOURCE_TRAIN_FILE --tgt-val-path $TARGET_TRAIN_FILE


# python run/main.py --src-train-path $SOURCE_TRAIN_FILE --tgt-train-path $TARGET_TRAIN_FILE --vocab-path $VOCAB_FILE --expt-dir $EXPT_PATH --load-checkpoint $CHECKPOINT_DIR