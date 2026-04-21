import numpy as np
from policy_evaluation import policy_evaluation

def policy_improvement(V, P, R, gamma):
    policy = []
    stable = True

    for s in range(len(P)):
        Q = []
        for a in range(len(P[s])):
            q = sum(P[s, a, s_next] *
                    (R[s, a] + gamma * V[s_next])
                    for s_next in range(len(P)))
            Q.append(q)

        best_action = int(np.argmax(Q))
        policy.append(best_action)

    return policy


def policy_iteration(P, R, gamma, theta):
    policy = [1, 1, 1]  # all Wait
    history_V = []
    history_policy = []

    while True:
        V = policy_evaluation(P, R, policy, gamma, theta)
        new_policy = policy_improvement(V, P, R, gamma)

        history_V.append(V.copy())
        history_policy.append(policy.copy())

        print("Policy:", new_policy, "V:", V)

        if new_policy == policy:
            break

        policy = new_policy

    return policy, V, history_V, history_policy