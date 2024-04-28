import numpy as np
from algorithms import invasive_weed


"""
CSCI 633: Biologically-Inspired Intelligent Systems

@author: Danny Gardner      drg5567
@author: Corey Urbanke      cju8676

Final Project, Invasive Weed Optimization with Differential Evolution
"""


rand_seed = 1990  # 123 #1234 #69
np.random.seed(rand_seed)

##############################################################################
# Set up algorithm hyper-parameters and settings
##############################################################################
D = 30                  # dimensionality of solution space for this problem
max_population = 50
seed_max = 5
seed_min = 2
n = 3                   # non-linear index
init_st_dev = .001
final_st_dev = 10
upper_bound = None      # TODO: replace values in limit
lower_bound = None
F = .5                  # differential weight
cr = .8                 # crossover rate


##############################################################################
# Set up Problems 1-3 of Bin Packing from the IWO-DE paper
##############################################################################
class PaperExperiment:
    def __init__(self, num_items, capacity, item_weights, iter_max):
        # n - number of items
        self.num_items = num_items
        # C - fixed capacity of a bin
        self.capacity = capacity
        # w_j - weights of items
        self.item_weights = item_weights
        # iter_max - maximum number of iterations
        self.iter_max = iter_max


# 5.1) Bin Packing Problem 1
exp_5_1 = PaperExperiment(30, 30, np.array([
    6, 3, 4, 6, 8, 7, 4, 7, 7, 5, 5, 6, 7, 7, 6, 4, 8, 7, 8, 8, 2, 3, 4, 5, 6, 5, 5, 7, 12
]), 500)

# 5.2) Bin Packing Problem 2
exp_5_2 = PaperExperiment(50, 30, np.array([
    6, 3, 4, 6, 8, 7, 4, 7, 7, 5, 5, 6, 7, 7, 6, 4, 8, 7, 8, 8, 2, 3, 4, 5, 6,
    5, 5, 7, 7, 12, 4, 6, 7, 8, 4, 5, 4, 8, 11, 4, 7, 4, 7, 6, 6, 8, 4, 5, 9, 10
]), 1000)

# 5.3) Bin Packing Problem 3
exp_5_3 = PaperExperiment(80, 50, np.array([
    6, 7, 3, 8, 10, 11, 12, 6, 4, 7, 8, 4, 3, 6, 7, 7, 5, 4, 6, 8, 9, 10, 11, 5, 6, 7, 7, 4,
    5, 3, 7, 8, 4, 5, 8, 9, 4, 5, 6, 12, 5, 3, 4, 5, 6, 5, 5, 7, 7, 12, 5, 8, 6, 8, 3, 5, 5,
    6, 6, 8, 3, 5, 6, 8, 9, 11, 12, 5, 6, 7, 5, 6, 8, 4, 5, 9, 10, 11, 4, 5
]), 2000)

paper_experiments = [exp_5_1, exp_5_2, exp_5_3]
##############################################################################
# Adapt to Knapsack problem for our reproducibility study
##############################################################################
# TODO for our knapsack problem use the papers weights and add a value foreach item


def main():

    # TODO run the papers experiments
    for exp in paper_experiments:
        print("Running experiment with {} items, capacity {}, and {} iterations".format(exp.num_items, exp.capacity, exp.iter_max))
        # TODO run the experiment
        invasive_weed(exp, max_population, seed_max, seed_min, n, init_st_dev, final_st_dev)

    # TODO run our experiments

    return


if __name__ == '__main__':
    main()
