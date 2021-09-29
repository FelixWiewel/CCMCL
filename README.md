# Condensed Composite Memory Continual Learning

This repository contains all the code used for the creation of the paper "Condensed Composite Memory Continual Learning" published at IJCNN 2021 (https://ieeexplore.ieee.org/document/9533491).

## Requirements

In order to install all required packages automatically run `pip install -r requirements.txt`.

## Running the code

Each experiment for a certain method and dataset can be run using the corresponding python script. Results will be saved in the `logs` folder.

### Experience Replay
- MNIST dataset:        `python run_experience_replay_MNIST.py`
- FashionMNIST dataset: `python run_experience_replay_FashionMNIST.py`
- SVHN dataset:         `python run_experience_replay_SVHN.py`
- CIFAR10 dataset:      `python run_experience_replay_CIFAR10.py`

### BiC
- MNIST dataset:        `python run_BiC_MNIST.py`
- FashionMNIST dataset: `python run_BiC_FashionMNIST.py`
- SVHN dataset:         `python run_BiC_SVHN.py`
- CIFAR10 dataset:      `python run_BiC_CIFAR10.py`

### Dataset Condensation
- MNIST dataset:        `python run_compressed_buffer_MNIST.py`
- FashionMNIST dataset: `python run_compressed_buffer_FashionMNIST.py`
- SVHN dataset:         `python run_compressed_buffer_SVHN.py`
- CIFAR10 dataset:      `python run_compressed_buffer_CIFAR10.py`

### Condensed Composite Memory Continual Learning
- MNIST dataset:        `python run_compositional_buffer_MNIST.py`
- FashionMNIST dataset: `python run_compositional_buffer_FashionMNIST.py`
- SVHN dataset:         `python run_compositional_buffer_SVHN.py`
- CIFAR10 dataset:      `python run_compositional_buffer_CIFAR10.py`
