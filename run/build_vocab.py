import argparse
import logging
import sys
sys.path.append('./')
from seq2seq.dataset.vocab import VietVocab
import logging
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-file', action='store', dest='source_file', 
                    help='Path to source data for build vocab')
    
    parser.add_argument('--save-file', action='store', dest='save_file',
                    help='Path to save_file for saving the vocab')
    
    parser.add_argument('--min-vocab', action='store', dest='min_vocab', default=1,
                    help='Min frequency of each vocab in vocab list.')

    
    opt = parser.parse_args()
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    logging.info(opt)


    vi_vocab = VietVocab(source_file=opt.source_file)
    vi_vocab.build_vocab(min_freq=opt.min_vocab)
    vi_vocab.save_vocab(save_file=opt.save_file)
    
    logger.info(f"Vocab = {vi_vocab.get_vocab().get_itos()}")