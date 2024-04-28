import numpy as np
import pandas as pd
from threading import Thread
import queue


def invasive_weed(exp, max_pop_size, seed_max, seed_min, n, init_st_dev, final_st_dev):
    # Initialize population
    init_pop_size = max_pop_size // 10
    weeds = np.random.randint(2, size=(init_pop_size, exp.num_items, exp.num_items))
    fitnesses = []

    min_fit = 0
    max_fit = 0
    step = 0
    while step < exp.iter_max:
        for i in range(init_pop_size):
            # seed propagation
            cur_fit = 0
            num_seeds = seed_propagation(cur_fit, min_fit, max_fit, seed_max, seed_min)

        # spatial diffusion distribution
        st_dev = spatial_distribution(exp.iter_max, step, n, init_st_dev, final_st_dev)

        # selection
        if init_pop_size == max_pop_size:
            pass

        step += 1
    return


def seed_propagation(cur_fit, min_fit, max_fit, seed_max, seed_min):
    term1 = (cur_fit - min_fit) / (max_fit - min_fit)
    term2 = seed_max - seed_min
    return seed_max - term1 * term2


def spatial_distribution(max_steps, step_num, n, init_st_dev, final_st_dev):
    term1 = ((max_steps - step_num) ** n) / max_steps ** n
    term2 = (init_st_dev - final_st_dev)
    return term1 * term2 + final_st_dev


def differential_evolution(f, D, pop_size, lower_bound, upper_bound, weight, cross_prob, num_threads):
    """
    Performs the differential evolution algortihm utilizing threading to minimize the value of a given function
    :param f: the objective function
    :param D: the dimension of the data
    :param pop_size: size of the population
    :param lower_bound: lower bound of values
    :param upper_bound: upper bound of values
    :param weight: the given weight for the donor vector
    :param cross_prob: the probability of crossover occuring
    :param num_threads: the number of threads to use
    :return: the best agent in the population with their fitness value
    """
    # TODO: determine what changes (if any) need to be made from this implementation to work for this problem
    # Initialize population
    population = pd.DataFrame(columns=["agent", "fitness"])

    for i in range(pop_size):
        x = (upper_bound - lower_bound) * np.random.rand(1, D) + lower_bound
        x_f = f(x)
        row = {"agent": x, "fitness": x_f}
        population.loc[len(population)] = row

    threads = []
    num_agents = pop_size // num_threads
    sub_pop_dataframes = queue.Queue()

    # Split the agents in the population into subpopulations based on the number of threads
    for i in range(num_threads):
        if i < num_threads - 1:
            sub_pop = population.iloc[i * num_agents: (i + 1) * num_agents].copy()
        else:
            sub_pop = population.iloc[i * num_agents:].copy()

        # Have each thread perform differential evolution on their subpopulations
        sub_pop = sub_pop.reset_index(drop=True)
        thread = Thread(target=parallel_de,
                        args=(f, D, sub_pop, weight, cross_prob, lower_bound, upper_bound, sub_pop_dataframes))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    # Regroup the subpopulations back into one population
    population = pd.DataFrame(columns=["agent", "fitness"])
    while not sub_pop_dataframes.empty():
        sub_df = sub_pop_dataframes.get()
        dfs_to_concat = [df for df in [population, sub_df] if not df.empty]
        population = pd.concat(dfs_to_concat, ignore_index=True)

    best_solution, best_fitness = find_best_de(population)
    return best_solution, best_fitness


# def mutate_and_cross(f, D, sub_pop, weight, cross_prob, lower_bound, upper_bound, result_queue):
def parallel_de(f, D, sub_pop, weight, cross_prob, lower_bound, upper_bound, result_queue):
    """
    Executes the differential evolution algorithm on a subset of the main population and adds the results to a queue
    :param f: the objective function
    :param D: the dimensions of each agent
    :param sub_pop: the subset of population to use
    :param weight: the given weight for the donor vector
    :param cross_prob: the probability of crossover
    :param lower_bound: lower bound of data values
    :param upper_bound: upper bound of data values
    :param result_queue: the queue to store the results
    :return: None
    """
    max_steps = 300
    step = 0
    pop_size = len(sub_pop)
    while step < max_steps:
        for i in range(pop_size):
            x_vec = sub_pop.loc[i, "agent"]
            x_fit = sub_pop.loc[i, "fitness"]
            donor = np.clip(gen_donor_vector(sub_pop, i, weight), lower_bound, upper_bound)

            rand_idx = np.random.randint(low=0, high=D)
            rand_val = np.random.random()

            cross_vec = x_vec
            for j in range(D):
                if rand_val <= cross_prob or j == rand_idx:
                    cross_vec[0][j] = donor[0][j]

            cross_fit = f(cross_vec)
            if cross_fit <= x_fit:
                sub_pop.loc[i] = {"agent": cross_vec, "fitness": cross_fit}
        step += 1
    result_queue.put(sub_pop)


def find_best_de(population):
    """
    Find the best agent in a population according to their fitness
    :param population: the population of agents
    :return: the best agent and their respective fitness
    """
    min_fit_idx = population["fitness"].idxmin()
    best_agent = population.loc[min_fit_idx, "agent"]
    best_fitness = population.loc[min_fit_idx, "fitness"]

    return best_agent, best_fitness


def gen_donor_vector(population, index, weight):
    """
    Generate a donor vector to use for mutation
    :param population: the population of agents to use
    :param index: the index of the current agent we are making a donor for
    :param weight: the given weight of the donor
    :return: the calculated donor vector
    """
    xp = None
    xq = None
    xr = None

    while xp is None or xq is None or xr is None:
        rand_idx = np.random.randint(low=0, high=len(population) - 1)
        if rand_idx == index:
            continue
        elif xp is None:
            xp = population.loc[rand_idx, "agent"]
        elif xq is None:
            xq = population.loc[rand_idx, "agent"]
        elif xr is None:
            xr = population.loc[rand_idx, "agent"]

    donor = xp + weight * (xq - xr)
    return donor
