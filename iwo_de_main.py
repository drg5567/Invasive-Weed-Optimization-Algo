import numpy as np


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
upper_bound = None        # TODO: replace values in limit
lower_bound = None
F = .5                  # differential weight
cr = .8                 # crossover rate


def bin_packing():
    # TODO: build the fitness function
    # sum number of boxes used
    # check loads of boxes to ensure they're not over capacity
    # check that items were only used once
    return


def main():
    return


if __name__ == '__main__':
    main()
