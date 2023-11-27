from __future__ import print_function, division

import torch
import torchtext
from torch import nn
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss, batch_size=64):
        self.loss = loss
        self.batch_size = batch_size
        self.pad_idx = None

    def evaluate(self, model, iterator):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (models): model to evaluate
            iterator (torch.utils.data.DataLoader): iterator to evaluate against
            
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
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

                loss = self.loss.criterion(output, trg)

                epoch_loss += loss.item()
                
                # Calculate accuracy
                # predictions = torch.argmax(output, dim=1)
                # correct_predictions += torch.sum(predictions == trg).item()
                # total_predictions += trg.numel()
                # Ignore padding tokens
                non_padding_tokens = (trg != self.pad_idx)
                predictions = torch.argmax(output, dim=1)

                correct_predictions += torch.sum(predictions[non_padding_tokens] == trg[non_padding_tokens]).item()
                total_predictions += torch.sum(non_padding_tokens).item()
                
        # accuracy = correct_predictions / total_predictions
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        # return self.loss.get_loss()
        return epoch_loss / len(iterator), accuracy