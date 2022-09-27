'''
This module contains methods for node classifiction
Some of the methods are also suitable for node clustering

Usage:
    - generate an instance of the learning method and specify the number of classes
    - use instance.estimate_labels(...) to estimate the labels for all nodes of the graph
'''

from src.node_classification._spectral_learning import SpectralLearning
from src.node_classification._lsbm_map import LsbmMap
from src.node_classification._lsbm_ml_lelarge import LsbmMlLelarge