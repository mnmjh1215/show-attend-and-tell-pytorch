# main.py
from config import Config
from train import Trainer
from utils import *
from model import Encoder, Decoder
from dataloader import DataLoader

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    device = Config.device
    print("PyTorch running with device {0}".format(device))

    if args.download:
        print("Downloading data")
        download_required_data()

    print("Generating word2id")
    word2id = generate_word2id('data/Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt')
    id2word = dict([(v, k) for k, v in word2id.items()])

    print("Loading Encoder and Decoder")
    encoder = Encoder(Config.encoded_size, Config.encoder_finetune)
    decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim,
                      vocab_size=len(word2id), dropout=Config.dropout, embedding_finetune=Config.embedding_finetune)


    if args.model_path:
        print("Loading model from model_path")
        load_model(encoder, decoder, args.model_path)
    else:
        # no model path, so load pretrained embedding
        print("Generating embedding from pretrained embedding file")
        embedding = load_pretrained_embedding('data/glove.6B.{}d.txt'.format(Config.embed_dim), word2id, Config.embed_dim)
        decoder.load_embedding(embedding)


    if not args.test:
        # train
        print("Loading DataLoader and Trainer")
        dloader = DataLoader('data/Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt', 'data/Flickr_Data/Images')
        trainer = Trainer(encoder, decoder, dloader)

        print("Start Training")
        loss_history = trainer.train(Config.num_epochs)
        plt.plot(np.arange(len(loss_history)), loss_history, label='Loss')
        plt.legend()
        plt.show()

    else:
        # test
        assert args.image_path

        encoder.eval()
        decoder.eval()

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()

        ])

        image = transform(Image.open(args.image_path))
        image = image.unsqueeze(0)

        # TODO
        # generate caption from an image
        encoder_output = encoder(image)
        captions, alphas = decoder.generate_caption_greedily(encoder_output)

        caption_in_word = ' '.join(list(map(id2word.get, captions[1:])))
        plt.imshow(image[0].numpy().transpose(1, 2, 0))
        plt.title(caption_in_word)
        plt.axis('off')
        plt.show()

        print(caption_in_word)


if __name__ == '__main__':
    args = get_args()
    main(args)
