from matplotlib import pyplot as plt


def plot_results(results, problem, exp_num):
    """
    Plot optimization line graph
    :param results: list of lists of (x, y) tuples to plot where x is iteration number and y is the minimum boxes found
    :param problem: name of the problem optimized
    :param exp_num: the current experiment number
    :return: N/A - displays plot
    """
    exp_num = exp_num + 1
    plt.figure()
    for result_tuple in results:
        result_list = result_tuple[0]
        name = result_tuple[1]
        x_vals, y_vals = zip(*result_list)
        plt.plot(x_vals, y_vals, label=name)
    plt.xlabel("Iteration")
    if problem == "Bin Packing":
        plt.ylabel("Minimum Boxes")
        plt.title("Optimization Results for Bin Packing Problem {}".format(exp_num))
        plt.legend()
        plt.savefig("binpacking_results_" + str(exp_num) + ".png")
    else:
        plt.ylabel("Maximum Value")
        plt.title("Optimization Results for Knapsack Problem {}".format(exp_num))
        plt.legend()
        plt.savefig("knapsack_results_" + str(exp_num) + ".png")
    return
