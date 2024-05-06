import numpy as np

"""
CSCI 633: Biologically-Inspired Intelligent Systems

@author: Danny Gardner      drg5567
@author: Corey Urbanke      cju8676

Objective functions for the Invasive Weed Optimization with Differential Evolution project.
"""


def one_d_bin_packing(x, exp):
    """
    Objective function for the one-dimensional bin packing problem.
    :param x: multidimensional array representing the solution of j items in i boxes
    :param exp: experiment object containing total n, C, w_j, and max_iter
    """

    y = np.sum(x, axis=1)
    y = np.where(y > 0, 1, 0)
    # calc the number of boxes used
    z = np.sum(y)
    # check if every item is in a box
    if np.any(np.sum(x, axis=0) != 1):
        return np.inf
    dot_product = np.dot(x, exp.item_weights)
    # check if each box is within capacity
    if np.any(dot_product > exp.capacity):
        return np.inf
    return z


def knapsack(x, exp):
    """
    Objective function for the knapsack problem.
    :param x: binary array representing solution of whether item is in knapsack
    :param exp: experiment object containing total n, C, w_j (weights of j items), v_j (values of j items), and max_iter
    :return: value of the knapsack given the solution
    """
    # calculate the total value of the knapsack
    value = np.dot(x, exp.item_values)
    # calculate the total weight of the knapsack
    weight = np.dot(x, exp.item_weights)
    # if the weight is over the capacity, return infinity
    if weight > exp.capacity:
        return -1
    return value
