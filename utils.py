import torch
import torch.nn as nn
import argparse
from config import Config
import torchvision.utils as tvutils
import matplotlib.pyplot as plt
import numpy as np


def generate_word2id(caption_file):
    """
    returns a dictionary, contatining (word, idx) pairs for every word used in caption file
    """
    word2id = {'<start>': 0, '.': 1}  # . serves as <end> token, so make it special
    idx = 2
    with open(caption_file) as fr:
        for line in fr:
            caption = line.lower().strip().split()[1:]
            for word in caption:
                if word not in word2id:
                    word2id[word] = idx
                    idx += 1
    word2id['<unk>'] = idx  # add unkown, just in case
    return word2id


def load_pretrained_embedding(pretrained_path, word2id):
    with open(pretrained_path) as pretrained:
        # TODO
        pass


def load_model(encoder, decoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])


def get_args():
    parser = argparse.ArgumentParser()

    # TODO
    args = parser.parse_args()

    return args
