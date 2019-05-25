
# _**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**_ in PyTorch

This repository contains PyTorch implementation of [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)

### How to run

To train model form scratch, use following command.

```
python main.py
```

To train model following existing checkpoint, use following command.

```
python main.py --model_path MODEL_PATH
```

To generate caption of an image, use following command.

```
python main.py --test --model_path MODEL_PATH --image_path IMAGE_PATH
```

Lastly, to download required data (Flickr8k and GloVe, for now), use '--download' argument.



## Results

Under training...