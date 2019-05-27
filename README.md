
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

### Flickr8k dataset

Following examples are generated after training using Google Colaboratory for less than 7 hours. Training captions are lemmatized, thus so is generated captions. Therefore generated captions are not complete English sentence, but they are still interpretable. (Lemmatization helps training when resource is limited, because it reduces vocabulary size.)

* Correct examples

![correct1](./images/Flickr8k/correct1.png)

![correct2](./images/Flickr8k/correct2.png)

![correct3](./images/Flickr8k/correct3.png)

![correct4](./images/Flickr8k/correct4.png)

![correct5](./images/Flickr8k/correct5.png)

![correct6](./images/Flickr8k/correct6.png)

![correct7](./images/Flickr8k/correct7.png)

* Not 100% correct, but not totally wrong examples

![not_correct1](./images/Flickr8k/not_correct.png)

![not_correct2](./images/Flickr8k/not_correct2.png)

![not_correct3](./images/Flickr8k/not_correct3.png)

* Wrong examples

![wrong](./images/Flickr8k/wrong.png)

![wrong2](./images/Flickr8k/wrong2.png)
