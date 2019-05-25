# Endocder, Attention and Decoder
# Show, Attend and Tell

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tvmodels
from config import Config


class Encoder(nn.Module):
    """
    An encoder that encodes each input image to tensor with shape (L, D)
    resnet101 is used for pretrained convolutional network.
    """
    def __init__(self, encoded_size=14, encoder_finetune=False):
        """

        :param encoded_size: size of image after being encoded.
        :param allow_finetune: if allow finetune, then encoder conv network is also trained.
        """
        super().__init__()

        self.encoded_size = encoded_size

        resnet101 = tvmodels.resnet101(pretrained=True)
        layers_to_use = list(resnet101.children())[:-2]
        # last two layers are AdaptiveAvgPool and Linear, which we don't need

        self.conv_net = nn.Sequential(*layers_to_use)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        if not encoder_finetune:
            for param in self.conv_net.parameters():
                param.requires_grad = False

    def forward(self, images):
        """

        :param images: Tensor with shape (batch_size, 3, image_size, image_size)
        :return:
        """
        x = self.conv_net(images)  # (batch_size, encoder_dim, image_size/32, image_size/32
        x = self.adaptive_pool(x)  # (batch_size, encoder_dim, self.encoded_size, self.encoded_size
        x = x.permute(0, 2, 3, 1)  # (batch_size, self.encoded_size, self.encoded_size, encoder_dim
        # since shape (batch_size, self.encoded_size ** 2, encoder_dim) will be used in decoder, do permutation

        batch_size = x.shape[0]
        encoder_dim = x.shape[-1]
        x = x.view(batch_size, -1, encoder_dim)  # (batch_size, L, D)
        # each point l in encoded image has vector with D-dim that represents that point
        # self.encoded_size ** 2 will be L and encoder_dim will be D in original paper's notation
        return x


class Attention(nn.Module):
    """
    Deterministic "soft" attention, which is differentiable and thus can be learned by backpropagation
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """

        :param encoder_dim: feature size of encoded image = last dimension of encoder output.
        :param decoder_dim: dimension of decoder's hidden state
        :param attention_dim: size of attention network. does not affect output dimension
        """
        super().__init__()
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_output, decoder_hidden):
        encoder_att = self.encoder_attention(encoder_output)  # (batch_size, L, attention_dim)
        decoder_att = self.decoder_attention(decoder_hidden)  # (batch_size, attention_dim)
        encoder_plus_decoder_att = encoder_att + decoder_att.unsqueeze(1)  # (batch_size, L, attention_dim)
        attention = self.attention(F.relu(encoder_plus_decoder_att)).squeeze(2)  # (batch_size, L)
        alpha = self.softmax(attention)  # (batch_size, L)
        context_vector = (encoder_output * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        # sum(dim=1) means summing over L
        # context vector is z_hat in original paper, calculated from h_t-1, and encoder output a

        return context_vector, alpha  # keep alpha for visualization?


class Decoder(nn.Module):
    """
    Decoder with attention
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim, embed_dim, vocab_size,
                 dropout=0.5, embedding_finetune=True):
        """

        :param encoder_dim: feature size of encoded image = last dimension of encoder output.
        :param decoder_dim: dimension of decoder's hidden state
        :param attention_dim: size of attention network. does not affect output dimension
        :param embed_dim: dimension of word embedding
        :param vocab_size: size of vocabulary
        :param dropout: dropout rate
        """

        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTMCell(self.encoder_dim + self.embed_dim, self.decoder_dim, bias=True)
        # h and c are initialized from encoder output.
        # authors used MLP, for now, use single layer perceptron
        self.init_h = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.init_c = nn.Linear(encoder_dim, decoder_dim, bias=False)

        # deep output layers
        self.L_h = nn.Linear(decoder_dim, embed_dim, bias=False)
        self.L_z = nn.Linear(encoder_dim, embed_dim, bias=False)
        self.L_o = nn.Linear(embed_dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        if not embedding_finetune:
            # always set embedding_finetune == True when not using pretrained embeddings
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.embedding_finetune = embedding_finetune

    def load_embedding(self, embedding):
        """
        :param embedding: pretraiend embedding, like GloVe or word2vec. Tensor with shape (vocab_size, embed_dim)
        """
        self.embedding.from_pretrained(embedding, freeze=not self.embedding_finetune)

    def init_hidden_states(self, encoder_output):
        """
        method to initialize hidden states. must be run at beginning of any forward propagation
        :param encoder_output: encoded output directly from encoder. shape (batch_size, L, encoder_dim=D)
        :return: initialized h and c, using self.init_h and self.init_c.
        """
        mean_encoder_output = encoder_output.mean(dim=1)  # mean over L
        init_h = self.init_h(mean_encoder_output)
        init_c = self.init_c(mean_encoder_output)

        return init_h, init_c

    def deep_output_layer(self, embedded_caption, h, context_vector):
        """

        :param embedded_caption: embedded caption, a tensor with shape (batch_size, embed_dim)
        :param h: hidden state, a tensor with shape (batch_size, decoder_dim
        :param context_vector: context vector, a tensor with shape (batch_size, encoder_dim)
        :return: output
        """

        scores = self.L_o(self.dropout(embedded_caption + self.L_h(h) + self.L_z(context_vector)))
        return scores

    def forward(self, encoder_output, captions):
        """
        forward method to be used at training time, because it requires captions as input

        :param encoder_output: encoder output, a tensor with shape (batch_size, L, encoded_dim=D)
        :param captions: captions encoded, a tensor with shape (batch_size, max_caption_length)
                         ex. [<start>, w1, w2, ... , wn, <end>]
        :return: predictions, alphas maybe?
        """

        batch_size = encoder_output.shape[0]
        num_pixels = encoder_output.shape[1]
        max_caption_length = captions.shape[-1]

        predictions = torch.zeros(batch_size, max_caption_length - 1, self.vocab_size).to(Config.device)
        alphas = torch.zeros(batch_size, max_caption_length - 1, num_pixels)  # save attention

        embedded_captions = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)

        h, c = self.init_hidden_states(encoder_output)

        for t in range(max_caption_length - 1):  # don't need prediction when y_t-1 is <end> or '.'
            embedded_caption_t = embedded_captions[:, t, :]  # (batch_size, embed_dim)
            context_vector, alpha = self.attention(encoder_output, h)
            # context vector has size (batch_size, encoder_dim)
            h, c = self.lstm(torch.cat([embedded_caption_t, context_vector], dim=1),
                             (h, c))
            preds = self.deep_output_layer(embedded_caption_t, h, context_vector)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas

    def generate_caption_greedily(self, encoder_output):
        """
        greedily generate captions for encoded images.

        :param encoder_output: encoder output, a tensor with shape (batch_size, L, encoded_dim)
        :return: captions generated greedily
        """
        # TODO
        h, c = self.init_hidden_states(encoder_output)
        captions = [0]  # 0 is <start>
        alphas = []
        while captions[-1] != 1 and len(captions) < 30:  # 1 is '.'
            caption = captions[-1]
            embedded_caption = self.embedding(torch.LongTensor([caption]))  # (1, embed_dim)
            context_vector, alpha = self.attention(encoder_output, h)  # (1, encoder_dim)
            h, c = self.lstm(torch.cat([embedded_caption, context_vector], dim=1),
                             (h, c))
            preds = self.deep_output_layer(embedded_caption, h, context_vector)  # (1, vocab_size)
            next_word = int(torch.argmax(preds, dim=1, keepdim=True).squeeze())
            captions.append(next_word)
            alphas.append(alpha)

        return captions, alphas
            

