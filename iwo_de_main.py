import numpy as np
from algorithms import invasive_weed, sim_anneal, firefly
from funcs import one_d_bin_packing as solve
from setup import PaperExperiment, KnapsackExperiment
from funcs import one_d_bin_packing, knapsack
from plot_results import plot_binpacking_results

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
cr = .8                 # crossover rate
# F = .5                  # differential weight


##############################################################################
# Set up Problems 1-3 of Bin Packing from the IWO-DE paper
##############################################################################
class PaperExperiement:
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
    6, 3, 4, 6, 8, 7, 4, 7, 7, 5, 5, 6, 7, 7, 6, 4, 8, 7, 8, 8, 2, 3, 4, 5, 6, 5, 5, 7, 7, 12
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

# simulated annealing hyperparamers #TODO: move somewhere nice and optimize these accordingly
# num_trials = 30
# Set the final temperature T_f
T_f = 1e-10
# Experiment with different initial temperatures
init_temps = [100, 50, 10, 5, 1]
# Experiment with different step-sizes for the random walk

# firefly algorithm hyperparameters #TODO: move somewehere nice and optimize to problems accordingly
# max iterations for each trial
N = 100
pop_size = [5, 10, 15, 25, 50]
alpha = [0.6, 0.7, 0.8, 0.9]
beta = 0.7
# gamma anywhere from [0.001, 1000]
gamma = [.1, 1, 10, 100, 500]


def main():
    # TODO run the papers experiments
    for exp in paper_experiments:
        ############### Invasive Weed Optimization ################
        print("Running experiment with {} items, capacity {}, and {} iterations".format(exp.num_items, exp.capacity,
                                                                                        exp.iter_max))
        print("Running Base Invasive Weed Optimization")
        base_tuple = (False, None)
        best_solution, num_steps, weed_results_to_plot = invasive_weed(exp, max_population, seed_max, seed_min, n,
                                                                       init_st_dev, final_st_dev, base_tuple)
        print("Minimum boxes: " + str(best_solution))
        print("Number of steps: " + str(num_steps))
        #####################################################

        ############### Invasive Weed Optimization with DE ################
        print("Running Invasive Weed Optimization with Differential Evolution")
        de_tuple = (True, cr)
        best_solution, num_steps, weed_de_results_to_plot = invasive_weed(exp, max_population, seed_max, seed_min, n,
                                                                          init_st_dev, final_st_dev, de_tuple)
        print("Minimum boxes: " + str(best_solution))
        print("Number of steps: " + str(num_steps))
        #####################################################

        ############### Simulated Annealing ################
        # temp = 1  # this temp works pretty good
        # N = exp.iter_max
        # # simulated annealing for given trial
        # print("Running simulated annealing with t_0 {}, N {}".format(temp, N))
        # x_star, sa_res_to_plot = sim_anneal(one_d_bin_packing, exp, temp, T_f, N)
        # # calculate the objective function value at the solution
        # f_star = one_d_bin_packing(x_star, exp)
        # print(f_star)
        #####################################################

        ################# Firefly Algorithm ################
        # a = alpha[0]
        # b = beta
        # g = gamma[1]
        # pop = pop_size[-2]
        # print("Running firefly algorithm with alpha {}, beta {}, gamma {} pop {}".format(a, b, g, pop))
        # x_star_firefly, fa_res_to_plot = firefly(one_d_bin_packing, exp, pop, exp.iter_max, a, b, g, D)
        # f_star_firefly = one_d_bin_packing(x_star_firefly, exp)
        # print(f_star_firefly)
        #####################################################

        ############### Plot Results ################
        # TODO: add FA to this graph
        result_list = [(weed_results_to_plot, "IWO"), (weed_de_results_to_plot, "DE-IWO")]#, (sa_res_to_plot, "SA"),
                       # (fa_res_to_plot, "FA")]
        plot_binpacking_results(result_list)
        # plot_binpacking_results([fa_res_to_plot])
        ##############################################


if __name__ == '__main__':
    # x = np.array([[1, 0, 1],
    #               [0, 1, 0],
    #               [0, 0, 0]])
    #
    # # Array of weights for each item
    # w = np.array([2, 3, 4])
    #
    # # Capacity constraint for each box
    # C = 5
    #
    # exp = PaperExperiment(3, 5, w, 1000)
    # fitness = solve(x, exp)
    # print(fitness)

    # w = np.array([2, 3, 4])
    # v = np.array([1, 2, 3])
    # C = 5
    # x = np.array([0, 0, 1])
    # exp = KnapsackExperiment(3, 5, w, v, 1000)
    # fitness = knapsack(x, exp)
    # print(fitness)
    main()
