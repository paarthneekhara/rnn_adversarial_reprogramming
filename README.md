# Adversarial Reprogramming of Sequence Classification Neural Networks

Code for our paper "Adversarial Reprogramming of Sequence Classification Neural Networks"

## Requirements
- Python 2.7.6
- Pytorch 0.4
- [Pytorch-ignite][1]

## Datasets
Download the data from .. and unzip is ```data/``` in the project directory.

## Training the Victim Models
Train the victim-models using ```python train_classifier.py --dataset=<dataset_name> --classifier_type=<victim_model>``` where dataset_name can be "Names", "SubNames", "TwitterArabic",  "IMDB". You may create custom-dataloaders for your own dataset in datasets.py

## Training the Adversarial Program

### White-box
Use ```python train_adversarial_gumbel.py --base_dataset=<original_task_dataset> --dataset=<adversarial_task_dataset> --classifier_type=<victim_model>```. Here victim_model can be "charRNN" (LSTM), "biRNN" (bi-LSTM), "CNN" (CNN) depending on the victim_model being attacked.

### Black-box
Use ```python train_adversarial.py --base_dataset=<original_task_dataset> --dataset=<adversarial_task_dataset> --classifier_type=<victim_model>```. Here victim_model can be "charRNN" (LSTM), "biRNN" (bi-LSTM), "CNN" (CNN) depending on the victim_model being attacked.

Use ```--help``` for further configurable options during training.

The checkpoints and training logs are saved during both black-box and white-box training. Use ```visualize_logs.py``` to plot the accuracies and reward logged during training.

[1]:https://github.com/pytorch/ignite

