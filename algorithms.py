import math
import numpy as np
from funcs import one_d_bin_packing


def invasive_weed(exp, max_pop_size, seed_max, seed_min, n, init_st_dev, final_st_dev, de_tuple):
    results = []
    # Initialize population
    init_pop_size = max_pop_size // 10
    weeds = initialize_weeds(init_pop_size, exp)
    fitnesses = []
    for i in range(init_pop_size):
        fitnesses.append(one_d_bin_packing(weeds[i], exp))

    min_fit = min(fitnesses)
    max_fit = max(fitnesses)
    step_of_best_sol = 0
    step = 0
    while step < exp.iter_max:
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
                fitnesses.append(one_d_bin_packing(offspring, exp))

        # selection
        if len(weeds) > max_pop_size:
            sorted_weeds, sorted_fitnesses = sort_weeds(fitnesses, weeds)
            fitnesses = sorted_fitnesses[:max_pop_size]
            weeds = sorted_weeds[:max_pop_size]
            new_min_fit = min(fitnesses)
            if new_min_fit < min_fit:
                step_of_best_sol = step
            min_fit = new_min_fit
            max_fit = max(fitnesses)

        if de_tuple[0]:
            cr = de_tuple[1]
            for w in range(len(weeds)):
                cur_weed = weeds[w]
                mutation = alt_mutate_weed(weeds, w, exp)
                rand_idx = np.random.randint(exp.num_items)
                rand_vec = np.random.uniform(size=exp.num_items)
                crossover_weed = cur_weed.copy()
                for k in range(exp.num_items):
                    if rand_vec[k] <= cr or k == rand_idx:
                        crossover_weed[:, k] = mutation[:, k]
                cross_fit = one_d_bin_packing(crossover_weed, exp)
                cur_fit = fitnesses[w]
                if cross_fit <= cur_fit:
                    weeds[w] = crossover_weed
                    fitnesses[w] = cross_fit

            weeds, fitnesses = sort_weeds(fitnesses, weeds)
            new_min_fit = min(fitnesses)
            if new_min_fit < min_fit:
                step_of_best_sol = step
            min_fit = new_min_fit
            max_fit = max(fitnesses)

        results.append((step, min_fit))
        step += 1
    return min_fit, step_of_best_sol, results


def initialize_weeds(num_weeds, exp):
    weeds = np.zeros((num_weeds, exp.num_items, exp.num_items))
    for i in range(num_weeds):
        for j in range(exp.num_items):
            box_num = np.random.randint(exp.num_items)
            weeds[i, box_num, j] = 1

    return weeds


def make_offspring(weeds, idx, st_dev, exp):
    # Copy the parent weed
    offspring = weeds[idx].copy()
    # Generate a normal distribution over the boxes
    box_dist = np.random.normal(loc=0, scale=st_dev, size=exp.num_items)
    # Use the sigmoid function to turn the distribution into probabilities
    box_sigmoid = 1 / (1 + np.exp(-box_dist))
    rand_vals = np.random.uniform(size=exp.num_items)
    boxes_in_use = []
    for k in range(exp.num_items):
        if rand_vals[k] >= box_sigmoid[k]:
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

    return offspring


def sort_weeds(fitnesses, weeds):
    sorted_idxs = np.argsort(fitnesses)
    sorted_fitnesses = [fitnesses[i] for i in sorted_idxs]
    sorted_weeds = weeds[sorted_idxs]
    return sorted_weeds, sorted_fitnesses


def seed_propagation(cur_fit, min_fit, max_fit, seed_max, seed_min):
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
    term1 = ((max_steps - step_num) ** n) / max_steps ** n
    term2 = (init_st_dev - final_st_dev)
    return term1 * term2 + final_st_dev


def alt_mutate_weed(weeds, cur_idx, exp):
    xp = None
    xq = None
    xr = None

    while xp is None or xq is None or xr is None:
        rand_idx = np.random.randint(len(weeds))
        if rand_idx == cur_idx:
            continue
        if xp is None:
            xp = weeds[rand_idx]
        if xq is None:
            xq = weeds[rand_idx]
        if xr is None:
            xr = weeds[rand_idx]

    p_boxes = np.random.randint(exp.num_items / 2)
    q_boxes = np.random.randint(exp.num_items / 2)
    # r_boxes = exp.num_items - p_boxes - q_boxes

    mutation = np.zeros(weeds[cur_idx].shape)

    mutation[:, :p_boxes] = xp[:, :p_boxes]
    mutation[:, p_boxes: p_boxes + q_boxes] = xq[:, p_boxes: p_boxes + q_boxes]
    mutation[:, p_boxes + q_boxes:] = xr[:, p_boxes + q_boxes:]
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
    x0 = initialize_weeds(1, exp)
    # define the cooling schedule T -> alpha*T (where 0< alpha < 1)
    T = T0
    # keep track of the best solution found
    x_star = x0
    # time step t (iteration counter)
    t = 0
    x_t = x0[0]
    # results in tuples of (step, min_fit)
    results = []
    while T > T_f and t < N:
        # Apply a discrete move to generate neighbor solution instead of Gaussian
        x_t1 = make_offspring(np.expand_dims(x_t, axis=0), 0, .0001, exp)
        # Calculate change in objective function
        delta_f = f(x_t1, exp) - f(x_t, exp)
        # Accept the new solution if better
        if delta_f < 0 or np.exp(-delta_f/T) > np.random.rand():
            # print("lower f(x) found: ", f(x_t1))
            x_t = x_t1
            x_star = x_t1
        # If not improved
        else:
            # Generate a random number r
            r = np.random.rand()
            # Accept if p = exp(-delta_f/T) > r (Boltzmann distribution)
            if np.exp(-delta_f/T) > r:
                x_t = x_t1
        # Update the temperature
        T -= (T0 - T_f) / N
        # Increment the iteration counter
        t += 1
        results.append((t, f(x_star, exp)))
    return x_star, results


def firefly(f, exp, pop_size, max_iter, alpha, beta, gamma, D):
    """
    Firefly algorithm (FA) for optimization.
    :param f: objective function (fourpeak, eggcrate, etc.)
    :param pop_size: population size
    :param max_iter: number of iterations
    :param alpha: randomness term
    :param beta: attractiveness term
    :param gamma: light absorption coefficient
    :param D: dimensionality of solution space
    :return best: best solution found
    """
    # Generate an initial population of n fireflies x_i (i = 1, 2, ..., n)
    pop = initialize_weeds(pop_size, exp)
    # light intensity I_i at x_i is determined by f(x_i)
    I = [f(p, exp) for p in pop]
    x_star = np.array([pop[np.argmin(I)]])
    # while (t<MaxGeneration)
    t = 0
    # results in tuples of (step, min_fit)
    results = []
    while t < max_iter:
        # for i = 1 : n ( all n fireflies )
        for i in range(pop_size):
            # for j = 1 : n ( all n fireflies ) (inner loop)
            for j in range(pop_size):
                # if (I_i > I_j)
                if I[i] > I[j]:
                    # vary attractiveness with distance r vis exp[-gamma*r^2]
                    r = np.linalg.norm(pop[i] - pop[j])
                    beta_i = beta * np.exp(-gamma * r**2)

                    # Evaluate new solutions and update light intensity
                    x_ij = beta_i * (pop[j] - pop[i]) + alpha * (np.random.rand(exp.num_items) - 0.5)
                    x_ij = x_ij[0, :]
                    # sigmoid function for probability of bit being 1 (according to
                    # Sayadi MK, Ramezanian R, Ghaffari-Nasab N. A discrete firefly meta-heuristic with local
                    # search for makespan minimization in permutation flow shop scheduling problems
                    s_i = 1 / (1 + np.exp(-x_ij))
                    # randomly generate values to determine if bit should be 1 or 0
                    rand_vals = np.random.uniform(size=exp.num_items)
                    boxes_in_use = []
                    for k in range(exp.num_items):
                        if rand_vals[k] >= s_i[k]:
                            # Don't use this box
                            pop[i, k] = np.zeros(exp.num_items)
                        else:
                            boxes_in_use.append(k)
                    # Loop over all the items and find ones that aren't currently in a box
                    for m in range(exp.num_items):
                        if np.all(pop[i, :, m] == 0):
                            box_num = -1
                            while box_num == -1:
                                # Randomly generate an index of a box that is already being used
                                rand_idx = np.random.randint(low=0, high=exp.num_items)
                                if rand_idx in boxes_in_use:
                                    box_num = rand_idx

                            pop[i, box_num, i] = 1
                    I[i] = f(pop[i], exp)
                    # print(I)
                    # I[i] = f(np.array(pop[i]))
        # Rank fireflies by their light intensity and find current global best g_star
        best = pop[np.argmin(I)]
        if f(best, exp) < f(x_star, exp):
            x_star = best
        results.append((t, f(x_star, exp)))
        t += 1
    return x_star, results
