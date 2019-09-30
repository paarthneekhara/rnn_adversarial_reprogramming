## Adversarial Reprogramming of Text Classification Neural Networks

Published as a conference paper at EMNLP 2019. [Arxiv Paper](https://arxiv.org/abs/1809.01829)

### Abstract
*Adversarial Reprogramming has demonstrated success in utilizing pre-trained neural network classifiers for alternative classification tasks without modification to the original network. An adversary in such an attack scenario trains an additive contribution to the inputs to repurpose the neural network for the new classification task. While this reprogramming approach works for neural networks with a continuous input space such as that of images, it is not directly applicable to neural networks trained for tasks such as text classification, where the input space is discrete. In this work, we introduce a context-based vocabulary remapping model to reprogram neural networks trained on a specific sequence classification task, for a new sequence classification task desired by the adversary. We propose training procedures for this adversarial program in both white-box and black-box settings. We demonstrate the application of our model by adversarially repurposing various text-classification models including LSTM, bi-directional LSTM and CNN for alternate classification tasks.*

![Left: White-box attack. Right: Black-box attack](https://i.imgur.com/EkuUZwm.png)
Left: White-box attack. Right: Black-box attack
### Requirements
- Python 2.7.6
- Pytorch 0.4
- [Pytorch-ignite][1]

### Datasets
Download the data from [this google drive link][2] in the project directory and unzip it.

### Training the Victim Models
Train the victim models on the original tasks using ```python train_classifier.py --dataset=<dataset_name> --classifier_type=<victim_model>``` where dataset_name can be "Names" (Names-18), "SubNames" (Names-5), "TwitterArabic",  "IMDB". ```<victim_model>``` can be one of *"charRNN", "biRNN", "CNN"* for LSTM, bi-LSTM and CNN respectively.

You may create custom-dataloaders for your own dataset in datasets.py. Use ```python train.py --help``` for setting hyperparameters of training models.

### Training the Adversarial Program

#### White-box
Use ```python train_adversarial_gumbel.py --base_dataset=<original_task_dataset> --dataset=<adversarial_task_dataset> --classifier_type=<victim_model>```. Use ```python train_adversarial_gumbel.py --help``` to get a list of configurable options including hyperparameters.  Allowed options for ```<original_task_dataset>``` and  ```<adversarial_task_dataset>``` are *Names, SubNames, TwitterArabic, IMDB*. Allowed options for victim_model are *"charRNN", "biRNN", "CNN"* for LSTM, bi-LSTM and CNN respectively.

#### Black-box
Use ```python train_adversarial.py --base_dataset=<original_task_dataset> --dataset=<adversarial_task_dataset> --classifier_type=<victim_model>```. Use ```python train_adversarial.py --help``` to get a list of configurable options including hyperparameters. Allowed options for ```<original_task_dataset>``` and  ```<adversarial_task_dataset>``` are *Names, SubNames, TwitterArabic, IMDB*. Allowed options for victim_model are *"charRNN", "biRNN", "CNN"* for LSTM, bi-LSTM and CNN respectively.

Use ```--help``` for setting the hyperparameters for adversarial program training.

The checkpoints and training logs are saved during both black-box and white-box training. Use ```visualize_logs.py``` to plot the accuracies and reward logged during training.

### Reprodocing the results
Hyper-parameter details to reproduce the results can be found in the supplementary material of our paper.

![](https://i.imgur.com/hNqWHvO.png)

[1]:https://github.com/pytorch/ignite
[2]:https://drive.google.com/file/d/1W7bBiDfTaQBOQKs52lfkUXBDtMPvSGfr/view?usp=sharing
