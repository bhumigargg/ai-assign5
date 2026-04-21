import matplotlib.pyplot as plt

def plot_values(V, states):
    plt.bar(states, V)
    plt.title("State Values")
    plt.xlabel("States")
    plt.ylabel("Value")
    plt.show()


def plot_value_iteration(history, states):
    for i, s in enumerate(states):
        plt.plot([h[i] for h in history], label=s)

    plt.legend()
    plt.title("Value Iteration Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.show()