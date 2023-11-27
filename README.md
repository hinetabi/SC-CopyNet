# SC-CopyNet
Spelling error correction with Vietnamese

# Requirements
- Ubuntu version 20.04
- Python 3.8.18
- Torch 2.1.0+cu118s
# Install dependency
```bash
$ conda create -n seq2seq python=3.8
$ conda activate seq2seq
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
$ pip install -r requirements.txt
```

# Running instructions

## 1. Build vocab
`If vocab is existed, can be loaded by using file (do not need to build again).`
```bash
$ sh tests/scripts/test_build_vocab.sh
```
## 2. Trains
```bash
$ sh run_script/train.sh
```
## 3. Benchmark (developing)
Set up logging with [wandb.io](https://wandb.ai/site)
```bash
$ wandb login 5d8018cfea10e827e71a6c0ecf6247a443473c27
```
## 4. Interaction with model (developing)