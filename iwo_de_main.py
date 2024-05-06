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
D = 30  # dimensionality of solution space for this problem
max_population = 50
seed_max = 5
seed_min = 2
n = 3  # non-linear index
init_st_dev = .001
final_st_dev = 10
cr = .8  # crossover rate


# F = .5                  # differential weight


##############################################################################
# Set up Problems 1-3 of Bin Packing from the IWO-DE paper
##############################################################################


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
exp_k_1 = KnapsackExperiment(30, 30, np.array([
    6, 3, 4, 6, 8, 7, 4, 7, 7, 5, 5, 6, 7, 7, 6, 4, 8, 7, 8, 8, 2, 3, 4, 5, 6, 5, 5, 7, 7, 12
]), np.array([]), 500)

exp_k_2 = KnapsackExperiment(50, 30, np.array([
    6, 3, 4, 6, 8, 7, 4, 7, 7, 5, 5, 6, 7, 7, 6, 4, 8, 7, 8, 8, 2, 3, 4, 5, 6,
    5, 5, 7, 7, 12, 4, 6, 7, 8, 4, 5, 4, 8, 11, 4, 7, 4, 7, 6, 6, 8, 4, 5, 9, 10
]), np.array([]), 1000)

exp_k_3 = KnapsackExperiment(80, 50, np.array([
    6, 7, 3, 8, 10, 11, 12, 6, 4, 7, 8, 4, 3, 6, 7, 7, 5, 4, 6, 8, 9, 10, 11, 5, 6, 7, 7, 4,
    5, 3, 7, 8, 4, 5, 8, 9, 4, 5, 6, 12, 5, 3, 4, 5, 6, 5, 5, 7, 7, 12, 5, 8, 6, 8, 3, 5, 5,
    6, 6, 8, 3, 5, 6, 8, 9, 11, 12, 5, 6, 7, 5, 6, 8, 4, 5, 9, 10, 11, 4, 5
]), np.array([]), 2000)

exp_k_4 = KnapsackExperiment(15, 750, np.array([
    70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120
]), np.array([135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]), 500)

exp_k_5 = KnapsackExperiment(24, 6404180, np.array([
    382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150, 823460, 903959, 853665, 551830,
    610856, 670702, 488960, 951111, 323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684
]), np.array([
    825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457, 1679693, 1902996, 1844992, 1049289,
    1252836, 1319836, 953277, 2067538, 675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261
]), 1000)

knapsack_experiments = [exp_k_4, exp_k_5]

# simulated annealing hyperparameters #TODO: move somewhere nice and optimize these accordingly
# num_trials = 30
# Set the final temperature T_f
T_f = 1e-10
# Experiment with different initial temperatures
init_temps = [100, 50, 10, 5, 1]
# Experiment with different step-sizes for the random walk

# firefly algorithm hyperparameters #TODO: move somewhere nice and optimize to problems accordingly
# max iterations for each trial
N = 100
pop_size = [5, 10, 15, 25, 50]
alpha = [0.6, 0.7, 0.8, 0.9]
beta = 0.7
# gamma anywhere from [0.001, 1000]
gamma = [.1, 1, 10, 100, 500]


def main():
    for experiment_list, f in [(paper_experiments, one_d_bin_packing), (knapsack_experiments, knapsack)]:
        for exp in experiment_list:
            ############### Invasive Weed Optimization ################
            print("Running experiment with {} items, capacity {}, and {} iterations".format(exp.num_items, exp.capacity,
                                                                                            exp.iter_max))
            print("Running Base Invasive Weed Optimization")
            base_tuple = (False, None)
            best_solution, num_steps, weed_results_to_plot = invasive_weed(f, exp, max_population, seed_max,
                                                                           seed_min, n,
                                                                           init_st_dev, final_st_dev, base_tuple)
            print("Minimum boxes: " + str(best_solution))
            print("Number of steps: " + str(num_steps))
            #####################################################

            ############### Invasive Weed Optimization with DE ################
            print("Running Invasive Weed Optimization with Differential Evolution")
            de_tuple = (True, cr)
            best_solution, num_steps, weed_de_results_to_plot = invasive_weed(f, exp, max_population,
                                                                              seed_max, seed_min, n,
                                                                              init_st_dev, final_st_dev, de_tuple)
            print("Minimum boxes: " + str(best_solution))
            print("Number of steps: " + str(num_steps))
            #####################################################

            ############### Simulated Annealing ################
            temp = 1  # this temp works pretty good
            N = exp.iter_max
            # simulated annealing for given trial
            print("Running simulated annealing with t_0 {}, N {}".format(temp, N))
            x_star, sa_res_to_plot = sim_anneal(f, exp, temp, T_f, N)
            # calculate the objective function value at the solution
            f_star = one_d_bin_packing(x_star, exp)
            print(f_star)
            #####################################################

            ################# Firefly Algorithm ################
            a = alpha[0]
            b = beta
            g = gamma[1]
            pop = pop_size[-2]
            print("Running firefly algorithm with alpha {}, beta {}, gamma {} pop {}".format(a, b, g, pop))
            x_star_firefly, fa_res_to_plot = firefly(f, exp, pop, exp.iter_max, a, b, g, D)
            f_star_firefly = one_d_bin_packing(x_star_firefly, exp)
            print(f_star_firefly)
            #####################################################

            ############### Plot Results ################
            # TODO: add FA to this graph
            result_list = [(weed_results_to_plot, "IWO"), (weed_de_results_to_plot, "DE-IWO"), (sa_res_to_plot, "SA"),
                           (fa_res_to_plot, "FA")]
            plot_binpacking_results(result_list)
            # plot_binpacking_results([(weed_de_results_to_plot, "DE-IWO")])
            ##############################################


if __name__ == '__main__':
    main()
