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

    def __init__(self, encoder, decoder, dataloader):
        self.encoder = encoder.to(Config.device)
        self.decoder = decoder.to(Config.device)
        self.dataloader = dataloader

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=Config.encoder_lr)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=Config.decoder_lr)
        self.criterion = nn.CrossEntropyLoss().to(Config.device)

    def train(self, num_epochs):
        # TODO
        pass

    def train_step(self, images, captions):
        # TODO
        pass

    

