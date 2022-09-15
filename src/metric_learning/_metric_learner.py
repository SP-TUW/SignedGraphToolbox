from abc import abstractmethod

class MetricLearner:



    def __init__(self, num_classes=2, verbose=0, save_intermediate=False):
        self.num_classes = num_classes
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.intermediate_results = None

    @abstractmethod
    def estimate_labels(self, graph, labels=None, guess=None):
        pass