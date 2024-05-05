import math
import numpy as np
from funcs import one_d_bin_packing


def invasive_weed(exp, max_pop_size, seed_max, seed_min, n, init_st_dev, final_st_dev, de_tuple):
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
            F = de_tuple[1]
            cr = de_tuple[2]
            for w in range(len(weeds)):
                cur_weed = weeds[w]
                # mutation = gen_weed_donor(weeds, w, F)
                mutation = mutate_weed(weeds[w], exp)
                rand_idx = np.random.randint(exp.num_items)
                rand_vec = np.random.uniform(size=exp.num_items)
                crossover_weed = cur_weed.copy()
                for k in range(exp.num_items):
                    if rand_vec[k] <= cr or k == rand_idx:
                        crossover_weed[k] = mutation[k]
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

        step += 1
    return min_fit, step_of_best_sol


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


def gen_weed_donor(weeds, index, scaling_factor):
    r1 = None
    r2 = None
    r3 = None

    while r1 is None or r2 is None or r3 is None:
        rand_idx = np.random.randint(len(weeds))
        if rand_idx == index:
            continue
        elif r1 is None:
            r1 = weeds[rand_idx]
        elif r2 is None:
            r2 = weeds[rand_idx]
        elif r3 is None:
            r3 = weeds[rand_idx]

    donor = r3 + scaling_factor * (r1 - r2)
    return donor


def mutate_weed(weed, exp):
    box_mutations = np.random.randint(1, exp.num_items / 4)

    for i in range(box_mutations):
        box_idx = np.random.randint(exp.num_items)
        item_idx = np.random.randint(exp.num_items)
        cur_bit = weed[box_idx][item_idx]
        if cur_bit == 1:
            weed[box_idx][item_idx] = 0
        else:
            weed[box_idx][item_idx] = 1
    return weed
