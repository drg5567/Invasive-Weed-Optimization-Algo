import math
import numpy as np

from setup import PaperExperiment, KnapsackExperiment
import time


def invasive_weed(f, exp, max_pop_size, seed_max, seed_min, n, init_st_dev, final_st_dev, de_tuple):
    """
    Performs the invasive weed optimization algorithm on 1-D Bin Packing
    :param f: function to be optimized
    :param exp: the experiment object containing the original data
    :param max_pop_size: the maximum number of weed agents to generate
    :param seed_max: the maximum number of seeds a weed can produce
    :param seed_min: the minimum number of seeds a weed can produce
    :param n: the non-linear index
    :param init_st_dev: the initial value of the standard deviation
    :param final_st_dev: the final value of the standard deviation
    :param de_tuple: a tuple that determines if the algorithm is running the differential evolution variant
    :return: The optimal solution, the number of steps to find the solution, along with data for generating a graph
    """
    results = []
    # Initialize population
    init_pop_size = max_pop_size // 10
    weeds = initialize_pop(init_pop_size, exp)
    fitnesses = []
    for i in range(init_pop_size):
        fitnesses.append(f(weeds[i], exp))

    min_fit = min(fitnesses)
    max_fit = max(fitnesses)
    step_of_best_sol = 0
    step = 0
    time_per_iter = []
    while step < exp.iter_max:
        start = time.time()
        cur_len = len(weeds)
        # spatial diffusion distribution
        st_dev = spatial_distribution(exp.iter_max, step, n, init_st_dev, final_st_dev)
        for i in range(cur_len):
            # seed propagation
            cur_fit = fitnesses[i]
            num_seeds = seed_propagation(cur_fit, min_fit, max_fit, seed_max, seed_min)

            for j in range(num_seeds):
                offspring = make_offspring(weeds, i, st_dev, exp)
                weeds = np.concatenate((weeds, np.expand_dims(offspring, axis=0)), axis=0)
                fitnesses.append(f(offspring, exp))

        # selection
        if len(weeds) > max_pop_size:
            sorted_weeds, sorted_fitnesses = sort_weeds(fitnesses, weeds, exp)
            fitnesses = sorted_fitnesses[:max_pop_size]
            weeds = sorted_weeds[:max_pop_size]
            min_fit, max_fit, step_of_best_sol = set_max_and_min(min_fit, max_fit, exp, fitnesses,
                                                                 step, step_of_best_sol)

        # Differential Evolution Variant
        if de_tuple[0]:
            cr = de_tuple[1]
            for w in range(len(weeds)):
                cur_weed = weeds[w]
                mutation = mutate_weed(weeds, w, exp)

                rand_idx = np.random.randint(exp.num_items)
                rand_vec = np.random.uniform(size=exp.num_items)
                crossover_weed = cur_weed.copy()
                for k in range(exp.num_items):
                    if rand_vec[k] <= cr or k == rand_idx:
                        if isinstance(exp, PaperExperiment):
                            crossover_weed[:, k] = mutation[:, k]
                        else:
                            crossover_weed[k] = mutation[k]
                cross_fit = f(crossover_weed, exp)
                cur_fit = fitnesses[w]
                if (isinstance(exp, PaperExperiment) and cross_fit <= cur_fit) or (
                        isinstance(exp, KnapsackExperiment) and cross_fit >= cur_fit):
                    weeds[w] = crossover_weed
                    fitnesses[w] = cross_fit

            # Re-sort the weeds after the DE steps have been taken
            weeds, fitnesses = sort_weeds(fitnesses, weeds, exp)
            min_fit, max_fit, step_of_best_sol = set_max_and_min(min_fit, max_fit, exp, fitnesses,
                                                                 step, step_of_best_sol)

        if isinstance(exp, PaperExperiment):
            results.append((step, min_fit))
        else:
            results.append((step, max_fit))
        step += 1
        end = time.time()
        time_per_iter.append(end - start)
    if isinstance(exp, PaperExperiment):
        best_solution = min_fit
    else:
        best_solution = max_fit
    return best_solution, step_of_best_sol, results, np.mean(time_per_iter)


def set_max_and_min(min_fit, max_fit, exp, fitnesses, step, step_of_best_sol):
    if isinstance(exp, PaperExperiment):
        new_min_fit = min(fitnesses)
        if new_min_fit < min_fit:
            step_of_best_sol = step
        min_fit = new_min_fit
        max_fit = max(fitnesses)
    else:
        new_max_fit = max(fitnesses)
        if new_max_fit > max_fit:
            step_of_best_sol = step
        max_fit = new_max_fit
        min_fit = min(fitnesses)
    return min_fit, max_fit, step_of_best_sol


def initialize_pop(num_weeds, exp):
    """
    Creates the initial agents for either the bin packing
    or knapsack problem
    :param num_weeds: number of agents to create
    :param exp: the experiment object
    :return: the initial population
    """
    if isinstance(exp, PaperExperiment):
        weeds = np.zeros((num_weeds, exp.num_items, exp.num_items))
        for i in range(num_weeds):
            for j in range(exp.num_items):
                box_num = np.random.randint(exp.num_items)
                weeds[i, box_num, j] = 1
    else:
        weeds = []
        for i in range(num_weeds):
            rand = np.random.uniform(size=exp.num_items)
            weeds.append(np.where(rand > .5, 1, 0))
    return weeds


def gen_valid_sol(exp, sigmoid_i, pop_i):
    offspring = pop_i.copy()
    rand_vals = np.random.uniform(size=exp.num_items)
    if isinstance(exp, PaperExperiment):
        boxes_in_use = []
        for k in range(exp.num_items):
            if rand_vals[k] >= sigmoid_i[k]:
                # Don't use this box
                offspring[k] = np.zeros(exp.num_items)
            else:
                boxes_in_use.append(k)

        # Loop over all the items and find ones that aren't currently in a box
        for i in range(exp.num_items):
            if np.all(offspring[:, i] == 0):
                box_num = -1
                while box_num == -1:
                    # Randomly generate an index of a box that is already being used
                    rand_idx = np.random.randint(low=0, high=exp.num_items)
                    if rand_idx in boxes_in_use:
                        box_num = rand_idx

                offspring[box_num, i] = 1
    else:
        for i in range(exp.num_items):
            if rand_vals[i] >= sigmoid_i[i]:
                offspring[i] = 0
            else:
                offspring[i] = 1
    return offspring


def make_offspring(weeds, idx, st_dev, exp):
    # Generate a normal distribution over the boxes
    normal_dist = np.random.normal(loc=0, scale=st_dev, size=exp.num_items)
    # Use the sigmoid function to turn the distribution into probabilities
    normal_sigmoid = 1 / (1 + np.exp(-normal_dist))
    return gen_valid_sol(exp, normal_sigmoid, weeds[idx])


def sort_weeds(fitnesses, weeds, exp):
    """
    Sort the weed population according to their fitness value
    :param fitnesses: the list of fitness values
    :param weeds: the population of weed agents
    :param exp: the experiment object
    :return: the sorted lists
    """
    sorted_idxs = np.argsort(fitnesses)
    if isinstance(exp, KnapsackExperiment):
        sorted_idxs = sorted_idxs[::-1]
    sorted_fitnesses = [fitnesses[i] for i in sorted_idxs]
    sorted_weeds = weeds[sorted_idxs]
    return sorted_weeds, sorted_fitnesses


def seed_propagation(cur_fit, min_fit, max_fit, seed_max, seed_min):
    """
    Determines the number of seeds a given weed can produce
    :param cur_fit: the weed's fitness
    :param min_fit: the smallest fitness value in the population
    :param max_fit: the largest fitness value in the population
    :param seed_max: the maximum number of seeds a weed can produce
    :param seed_min: the minimum number of seeds a weed can produce
    :return: the number of seeds the weed can produce
    """
    if cur_fit == float('inf'):
        cur_fit = 1000
    if min_fit == float('inf'):
        min_fit = 1000
    if max_fit == float('inf'):
        max_fit = 1000

    term1 = (cur_fit - min_fit) / (max_fit - min_fit) if max_fit != min_fit else 0
    term2 = seed_max - seed_min
    num_seeds = math.floor(seed_max - term1 * term2)
    return num_seeds if num_seeds > 0 else 0


def spatial_distribution(max_steps, step_num, n, init_st_dev, final_st_dev):
    """
    Calculates the current standard deviation value for offspring generation based on the current step number
    :param max_steps: the maximum number of steps the algorithm can go
    :param step_num: the current step number
    :param n: the non-linear index
    :param init_st_dev: the initial standard deviation
    :param final_st_dev: the final standard deviation
    :return: the current standard deviation
    """
    term1 = ((max_steps - step_num) ** n) / max_steps ** n
    term2 = (init_st_dev - final_st_dev)
    return term1 * term2 + final_st_dev


def mutate_weed(weeds, cur_idx, exp):
    """
    Generates a donor matrix to be used in crossover during the differential evolution steps
    :param weeds: the population of weed agents
    :param cur_idx: the current index of the weed agent
    :param exp: the experiment object
    :return: a new mutated weed based on 3 randomly chosen weed agents
    """
    xp = None
    xq = None
    xr = None

    while xp is None or xq is None or xr is None:
        rand_idx = np.random.randint(len(weeds))
        if rand_idx == cur_idx:
            continue
        elif xp is None:
            xp = weeds[rand_idx]
        elif xq is None:
            xq = weeds[rand_idx]
        elif xr is None:
            xr = weeds[rand_idx]

    p_cols = np.random.randint(exp.num_items / 2)
    q_cols = np.random.randint(exp.num_items / 2)
    # r_cols = exp.num_items - p_cols - q_cols

    mutation = np.zeros(weeds[cur_idx].shape)

    if isinstance(exp, PaperExperiment):
        mutation[:, :p_cols] = xp[:, :p_cols]
        mutation[:, p_cols: p_cols + q_cols] = xq[:, p_cols: p_cols + q_cols]
        mutation[:, p_cols + q_cols:] = xr[:, p_cols + q_cols:]
    else:
        mutation[:p_cols] = xp[:p_cols]
        mutation[p_cols: p_cols + q_cols] = xq[p_cols:p_cols + q_cols]
        mutation[p_cols + q_cols:] = xr[p_cols + q_cols:]
    return mutation


def sim_anneal(f, exp, T0, T_f, N):
    """
    Simulated Annealing (SA) algorithm for optimization, discretized for our problems
    :param f: objective function (rosenbrock, ackley, etc.)
    :param exp: object containing num items, capacity, box weights, etc.
    :param T0: initial temperature
    :param T_f: final temperature
    :param N: number of iterations
    :return x_hat: best solution found
    """
    x0 = initialize_pop(1, exp)
    # define the cooling schedule T -> alpha*T (where 0< alpha < 1)
    T = T0
    # keep track of the best solution found
    x_star = x0[0]
    # time step t (iteration counter)
    t = 0
    x_t = x0[0]
    # results in tuples of (step, min_fit)
    results = []
    time_per_iter = []
    while T > T_f and t < N:
        start = time.time()
        # Apply a discrete move to generate neighbor solution instead of Gaussian
        x_t1 = make_offspring(np.expand_dims(x_t, axis=0), 0, .0001, exp)
        # Calculate change in objective function
        delta_f = f(x_t1, exp) - f(x_t, exp)
        if isinstance(exp, PaperExperiment):
            if delta_f < 0 or np.exp(-delta_f / T) > np.random.rand():
                x_t = x_t1
                x_star = x_t1
            else:
                r = np.random.rand()
                if np.exp(-delta_f / T) > r:
                    x_t = x_t1
        else:
            if delta_f > 0 or np.exp(delta_f / T) > np.random.rand():
                x_t = x_t1
                x_star = x_t1
            else:
                r = np.random.rand()
                if np.exp(delta_f / T) > r:
                    x_t = x_t1
        # Update the temperature
        T -= (T0 - T_f) / N
        # Increment the iteration counter
        t += 1
        end = time.time()
        time_per_iter.append(end - start)
        results.append((t, f(x_star, exp)))
    return x_star, results, np.mean(time_per_iter)


def firefly(f, exp, pop_size, max_iter, alpha, beta, gamma, D):
    """
    Firefly algorithm (FA) for optimization.
    :param f: objective function (fourpeak, eggcrate, etc.)
    :param exp: object containing num items, capacity, box weights, etc.
    :param pop_size: population size
    :param max_iter: number of iterations
    :param alpha: randomness term
    :param beta: attractiveness term
    :param gamma: light absorption coefficient
    :param D: dimensionality of solution space
    :return best: best solution found
    """
    # Generate an initial population of n fireflies x_i (i = 1, 2, ..., n)
    pop = initialize_pop(pop_size, exp)
    # light intensity I_i at x_i is determined by f(x_i)
    I = [f(p, exp) for p in pop]
    if isinstance(exp, PaperExperiment):
        x_star = pop[np.argmin(I)]
    else:
        x_star = pop[np.argmax(I)]
    # while (t<MaxGeneration)
    t = 0
    # results in tuples of (step, min_fit)
    results = []
    time_per_iter = []
    while t < max_iter:
        start = time.time()
        # for i = 1 : n ( all n fireflies )
        for i in range(pop_size):
            # for j = 1 : n ( all n fireflies ) (inner loop)
            for j in range(pop_size):
                # if (I_i > I_j)
                if I[i] > I[j]:
                    # vary attractiveness with distance r vis exp[-gamma*r^2]
                    r = np.linalg.norm(pop[i] - pop[j])
                    beta_i = beta * np.exp(-gamma * r ** 2)

                    # Evaluate new solutions and update light intensity
                    x_ij = beta_i * (pop[j] - pop[i]) + alpha * (np.random.rand(exp.num_items) - 0.5)
                    # sigmoid function for probability of bit being 1 (according to
                    # Sayadi MK, Ramezanian R, Ghaffari-Nasab N. A discrete firefly meta-heuristic with local
                    # search for makespan minimization in permutation flow shop scheduling problems
                    if isinstance(exp, PaperExperiment):
                        s_i = 1 / (1 + np.exp(-x_ij[0, :]))
                    else:
                        s_i = 1 / (1 + np.exp(-x_ij))
                    pop_i_temp = gen_valid_sol(exp, s_i, pop[i])

                    # update light intensity
                    if (isinstance(exp, PaperExperiment) and f(pop_i_temp, exp) != np.inf) or (
                            isinstance(exp, KnapsackExperiment) and f(pop_i_temp, exp) != -1):
                        I[i] = f(pop_i_temp, exp)
                        pop[i] = pop_i_temp

        # Rank fireflies by their light intensity and find current global best g_star
        if isinstance(exp, PaperExperiment):
            best = pop[np.argmin(I)]
            if f(best, exp) < f(x_star, exp):
                x_star = best
        else:
            best = pop[np.argmax(I)]
            if f(best, exp) > f(x_star, exp):
                x_star = best
        results.append((t, f(x_star, exp)))
        t += 1
        end = time.time()
        time_per_iter.append(end - start)
    return x_star, results, np.mean(time_per_iter)
