from abc import abstractmethod

class NodeLearner:



    def __init__(self, num_classes, verbosity=0, save_intermediate=False):
        self.num_classes = num_classes
        self.verbosity = verbosity
        self.save_intermediate = save_intermediate
        self.intermediate_results = None
        self.l_est = None
        self.embedding = None
        self.normalized_embedding = None

    @abstractmethod
    def estimate_labels(self, graph, labels=None, guess=None):
        pass