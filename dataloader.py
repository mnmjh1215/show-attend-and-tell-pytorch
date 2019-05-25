# for some reason, torchvision.datasets.Flickr8k doesn't work on my data,
# which was downloaded from https://www.kaggle.com/srbhshinde/flickr8k-sau
# so I made my own dataloader

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

from config import Config
from utils import generate_word2id


class DataLoader:
    def __init__(self, caption_file, image_path):
        self.word2id = generate_word2id(caption_file)

        self.data = {}
        with open(caption_file) as fr:
            for line in fr:
                line = line.lower().strip().split()
                image_filename = line[0].split('#')[0]
                image_filename = os.path.join(image_path, image_filename)
                if not os.path.isfile(image_filename):
                    continue

                # image exists
                caption = ['<start>'] + line[1:]  # add <start> token as the first token
                if caption[-1] != '.':
                    caption.append('.')  # add '.', which is the end of sentence, if it is not there

                caption_length = len(caption)
                if caption_length < 4:
                    # that is, other than <start> and '.', only one word is captioned
                    # indicating that this caption is a garbage data
                    continue
                if not self.data.get(caption_length):
                    self.data[caption_length] = {'images': [], 'captions': []}

                caption_tokenized = [self.word2id.get(word, self.word2id.get('<unk>')) for word in caption]
                self.data[caption_length]['images'].append(image_filename)
                self.data[caption_length]['captions'].append(caption_tokenized)

        for key, val in self.data.items():
            # convert lists to numpy arrays. numpy array is better to use than list.
            self.data[key]['captions'] = np.array(val['captions'])
            self.data[key]['images'] = np.array(val['images'])

        # transforms to be applied to images
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()

        ])

    def get_random_minibatch(self, batch_size=Config.batch_size):
        """
        Returns minibatch of images and captions that have same caption length
        This way, no computation time is wasted, and this is also the method authors of Show, Attend and Tell used.
        """
        caption_length = self.choose_caption_length()  # randomly choose caption length

        # returns min(bath_size, len(data[caption_length]['captions'])) data for randomly chosen caption_length
        data_len = len(self.data[caption_length]['images'])
        batch_size = min(batch_size, data_len)

        # random choices
        choices = np.random.choice(data_len, batch_size, replace=False)
        # load captions
        captions = torch.from_numpy(self.data[caption_length]['captions'][choices])

        # iteratively load images
        filenames = self.data[caption_length]['images'][choices]
        images = []
        for filename in filenames:
            images.append(self.transforms(Image.open(filename)))
        images = torch.stack(images)

        return images, captions

    def choose_caption_length(self, proportion_factor=0.8):
        # randomly choose caption length,
        # proportional to (number of data having the caption lengths) ** proportion_factor

        max_len = max(self.data)
        num_data_per_length = np.array([0] * (max_len + 1))
        for key in self.data:
            num_data_per_length[key] = len(self.data[key]['images']) ** proportion_factor

        prob = num_data_per_length / sum(num_data_per_length)

        choice = np.random.choice(max_len + 1, 1, p=prob)
        
        return int(choice)

    def __len__(self):
        length = 0
        for key in self.data:
            length += len(self.data[key]['images'])

        return length
