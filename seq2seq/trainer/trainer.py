from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint

import wandb

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir, loss=CrossEntropyLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=1000, print_every=100,):

        self._trainer = "Seq2Seq Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
        

    def _train_batch(self, model, src, trg, clip = 1):
        model.train()

        self.optimizer.optimizer.zero_grad()

        output = model(src, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = self.loss.criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        self.optimizer.step()

        return loss.item()

    def _train_epoches(self, train_iter, model, n_epochs, start_epoch, start_step, vi_vocab,
                       val_iter=None):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        steps_per_epoch = len(train_iter)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            # batch_generator = train_iter.__iter__()
            # consuming seen batches from previous training
            # for _ in range((epoch - 1) * steps_per_epoch, step):
                # next(batch_generator)

            model.train(True)
            for i, batch in enumerate(train_iter):
                
                step += 1
                step_elapsed += 1

                src = batch[0]
                trg = batch[1]
                
                loss = self._train_batch(model, src, trg, clip=1)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)
                    
                    wandb.log({"avg_train_loss": print_loss_avg})
                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=vi_vocab.vocab,
                               output_vocab=vi_vocab.vocab).save(self.expt_dir)

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if val_iter is not None:
                self.evaluator.pad_idx = vi_vocab['<pad>']
                dev_loss, acc = self.evaluator.evaluate(model, val_iter)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy %.4f" % (self.loss.name, dev_loss, acc)
                # log metrics to wandb
                wandb.log({"acc": acc, "avg_val_loss": dev_loss})
                
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, train_iter, vi_vocab, optimizer, num_epochs, val_iter,
              resume=False):
        """ Run training for a given model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(train_iter, model, num_epochs,
                            start_epoch, step, val_iter=val_iter,vi_vocab=vi_vocab)
                            
        return model