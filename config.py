# all configs are saved here
import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_avaialble() else "cpu")
    encoder_dim = 512
    decoder_dim = 512
    attention_dim = 512
    dropout = 0.5
    embed_dim = 300  # glove supports 50, 100, 200, 300
    encoder_lr = 1e-4
    decoder_lr = 1e-4

