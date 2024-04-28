class Experiment:
    def __init__(self, num_items, capacity, iter_max):
        # n - number of items
        self.num_items = num_items
        # C - fixed capacity of a bin
        self.capacity = capacity
        # iter_max - maximum number of iterations
        self.iter_max = iter_max

class PaperExperiment(Experiment):
    def __init__(self, num_items, capacity, item_weights, iter_max):
        super(PaperExperiment, self).__init__(num_items, capacity, iter_max)
        # w_j - weights of items
        self.item_weights = item_weights


class KnapsackExperiment(Experiment):
    def __init__(self, num_items, capacity, item_weights, item_values, iter_max):
        super(KnapsackExperiment, self).__init__(num_items, capacity, iter_max)
        # w_j - weights of items
        self.item_weights = item_weights
        # v_j - values of items
        self.item_values = item_values