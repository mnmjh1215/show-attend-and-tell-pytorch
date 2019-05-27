import torch
import torch.nn as nn
import argparse
from config import Config
import torchvision.utils as tvutils
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess


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


def load_pretrained_embedding(pretrained_path, word2id, embed_size=300):
    vocab_size = len(word2id)
    embedding = torch.randn((vocab_size, embed_size))
    with open(pretrained_path) as fr:
        # TODO
        for line in fr:
            line = line.lower().strip().split()
            word = line[0]
            vec = list(map(float, line[1:]))

            if word in word2id:
                idx = word2id[word]
                embedding[idx] = torch.Tensor(vec)

    return embedding


def load_model(encoder, decoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])


def download_required_data():
    # first, download Flickr dataset from kaggle
    if not os.path.isdir('data/'):
        os.mkdir('data/')
    subprocess.run(['kaggle', 'datasets', 'download', 'srbhshinde/flickr8k-sau', '-p', 'data/'])
    subprocess.run(['unzip', '-q', 'data/flickr8k-sau.zip', '-d', 'data/'])
    subprocess.run(['rm', 'data/flickr8k-sau.zip'])

    # next, download glove
    subprocess.run(['wget', 'nlp.stanford.edu/data/glove.6B.zip', '-P', 'data/'])
    subprocess.run(['unzip', '-q', 'data/glove.6B.zip', '-d', 'data/'])
    subprocess.run(['rm', 'data/glove.6B.zip'])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test',
                        action='store_true',
                        help='use this argument to generate caption')

    parser.add_argument('--image_path',
                        help='required for testing')

    parser.add_argument('--model_path',
                        help='path to save model. required for testing. '
                             'If not given in training, models will be trained from scratch')

    parser.add_argument('--download',
                        help='use this argument to download required files',
                        action='store_true')

    parser.add_argument('--lemmatize',
                        help='use this argument to use lemmatized training captions',
                        action='store_true')

    # TODO
    args = parser.parse_args()

    return args
