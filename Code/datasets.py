"""
This file contains data sets for continual learning.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import abc

# Disable progress bar
tfds.disable_progress_bar()


class DataSet(abc.ABC):
    # Base class for data set classes
    @abc.abstractmethod
    def __init__(self):
        pass

    def filter_fn(self, batch, classes):
        return tf.reduce_any(tf.math.equal(batch["label"], classes))

    def get_split(self, classes):
        train_data = self.train_data.filter(lambda x: self.filter_fn(x, classes))
        val_data = self.val_data.filter(lambda x: self.filter_fn(x, classes))
        test_data = self.test_data.filter(lambda x: self.filter_fn(x, classes))
        return train_data, val_data, test_data

    def get_all(self):
        return self.train_data, self.val_data, self.test_data


class SplitMNIST(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="mnist", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="mnist", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="mnist", split="test")


class SplitEMNIST(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="emnist/letters", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="emnist/letters", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="emnist/letters", split="test")


class SplitFashionMNIST(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="fashion_mnist", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="fashion_mnist", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="fashion_mnist", split="test")


class SplitCIFAR10(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="cifar10", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="cifar10", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="cifar10", split="test")


class SplitCIFAR100(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="cifar100", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="cifar100", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="cifar100", split="test")


class SplitOMNIGLOT(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="omniglot", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="omniglot", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="omniglot", split="test")


class SplitSVHN(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="svhn_cropped", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="svhn_cropped", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="svhn_cropped", split="test")


class SplitCaltech101(DataSet):
    def __init__(self, num_validation):
        self.train_data = tfds.load(name="caltech101", split="train[{:d}:]".format(int(num_validation)))
        self.val_data = tfds.load(name="caltech101", split="train[:{:d}]".format(int(num_validation)))
        self.test_data = tfds.load(name="caltech101", split="test")
