from mdp import build_mdp, STATES, GAMMA, THETA
from policy_evaluation import policy_evaluation
from value_iteration import value_iteration, extract_policy
from policy_iteration import policy_iteration
from plots import plot_values, plot_value_iteration

def main():
    P, R = build_mdp()

    print("Checking probabilities:")
    for s in range(3):
        for a in range(2):
            print(s, a, sum(P[s, a]))

    print("Rewards:\n", R)

    # Task 1: Policy Evaluation
    policy = [0, 1, 1]  # Search, Wait, Wait
    V = policy_evaluation(P, R, policy, GAMMA, THETA)
    print("Vπ:", V)
    plot_values(V, STATES)

    # Task 2: Value Iteration
    V_star, history = value_iteration(P, R, GAMMA, THETA)
    optimal_policy = extract_policy(V_star, P, R, GAMMA)

    print("Optimal V:", V_star)
    print("Optimal Policy:", optimal_policy)

    plot_value_iteration(history, STATES)

    # Task 3: Policy Iteration
    policy_opt, V_opt, hist_V, hist_policy = policy_iteration(P, R, GAMMA, THETA)

    print("Final Policy:", policy_opt)
    print("Final V:", V_opt)


if __name__ == "__main__":
    main()