# all configs are saved here
import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder
    encoded_size = 14  # size of encoded image
    encoder_finetune = False
    # decoder
    encoder_dim = 512
    decoder_dim = 512
    attention_dim = 512
    dropout = 0.5
    embed_dim = 300  # glove supports 50, 100, 200, 300
    embedding_finetune = True

    encoder_lr = 1e-4
    decoder_lr = 5e-4

    # dataloader
    batch_size = 32

    # train
    num_epochs = 25
