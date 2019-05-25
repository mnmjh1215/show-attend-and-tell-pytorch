# defines training procedure

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config


class Trainer:
    """
    Trainer for encoder and decoder
    """

    def __init__(self, encoder, decoder, dataloader, iter_per_epoch=None, batch_size=32):
        self.encoder = encoder.to(Config.device)
        self.decoder = decoder.to(Config.device)
        self.dataloader = dataloader

        encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
        if encoder_params:        
            self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                lr=Config.encoder_lr)
        else:
            self.encoder_optimizer = None
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=Config.decoder_lr)
        self.criterion = nn.CrossEntropyLoss().to(Config.device)

        self.loss_hist = []
        self.curr_epoch = 0

        if not iter_per_epoch:
            self.iter_per_epoch = len(dataloader) // 32
        else:
            self.iter_per_epoch = iter_per_epoch

        self.batch_size = batch_size

    def train(self, num_epochs):
        # TODO
        for epoch in range(self.curr_epoch, num_epochs):
            epoch_loss = 0

            start = time.time()
            for iter in range(self.iter_per_epoch):
                images, captions = self.dataloader.get_random_minibatch(self.batch_size)
                images = images.to(Config.device)
                captions = captions.to(Config.device)

                loss = self.train_step(images, captions)
                epoch_loss += loss

                if (iter + 1) % 100 == 0:
                    print("[{0}/{1}] [{2}/{3}] loss: {4:.4f}".format(epoch + 1, num_epochs, iter + 1, 
                                                                self.iter_per_epoch, epoch_loss / (iter + 1)))

            # end of epoch
            print("epoch {0} {1:.4f} seconds, loss: {2:.4f}".format(epoch + 1, time.time() - start,
                                                               epoch_loss / self.iter_per_epoch))
            self.curr_epoch += 1

        # end of training.
        # save checkpoint
        if not os.path.isdir('checkpoints/'):
            os.mkdir('checkpoints/')
        self.save_checkpoint('checkpoints/epoch-{0}.ckpt'.format(num_epochs))

        return self.loss_hist


    def train_step(self, images, captions):
        # TODO

        self.reset_grad()

        encoded_images = self.encoder(images)
        predictions, alphas = self.decoder(encoded_images, captions)
        
        # for calculating loss, first caption <start> is not necessary
        captions = captions[:, 1:]
        batch_size = captions.shape[0]
        caption_length = captions.shape[1]
        loss = self.criterion(predictions.view(batch_size * caption_length, -1), captions.contiguous().view(-1))
        loss.backward()
        
        if self.encoder_optimizer:
            self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        self.loss_hist.append(loss.item())

        return loss.item()

    def reset_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def save_checkpoint(self, checkpoint_path):
        if self.encoder_optimizer:
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
                'epoch': self.curr_epoch,
                'loss_hist': self.loss_hist,
            }, checkpoint_path)

        else:
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
                'epoch': self.curr_epoch,
                'loss_hist': self.loss_hist,
            }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
        if self.encoder_optimizer:
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        self.curr_epoch = checkpoint['epoch']
        self.loss_hist = checkpoint['loss_hist']

