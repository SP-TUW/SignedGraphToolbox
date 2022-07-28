from abc import ABC, abstractmethod

class NodeLearner:



    def __init__(self, num_classes, verbose=0, save_intermediate=False):
        self.num_classes = num_classes
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.intermediate_results = None
        self.l_est = None

    @abstractmethod
    def estimate_labels(self, data, labels=None, guess=None):
        pass