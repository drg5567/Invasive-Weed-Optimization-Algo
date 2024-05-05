from matplotlib import pyplot as plt


def plot_binpacking_results(results):
    """
    Plot optimization line graph
    :param results: list of lists of (x, y) tuples to plot where x is iteration number and y is the minimum boxes found
    :return: N/A - displays plot
    """
    plt.figure()
    for result_tuple in results:
        result_list = result_tuple[0]
        name = result_tuple[1]
        x_vals, y_vals = zip(*result_list)
        plt.plot(x_vals, y_vals, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Minimum Boxes")
    plt.title("Optimization Results")
    plt.legend()
    plt.show()
    return
