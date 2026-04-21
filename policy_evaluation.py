import numpy as np

def policy_evaluation(P, R, policy, gamma, theta):
    V = np.zeros(len(P))
    iteration = 0

    while True:
        delta = 0
        for s in range(len(P)):
            v = V[s]
            a = policy[s]

            V[s] = sum(P[s, a, s_next] * (R[s, a] + gamma * V[s_next])
                       for s_next in range(len(P)))

            delta = max(delta, abs(v - V[s]))

        iteration += 1
        if delta < theta:
            break

    print(f"Policy Evaluation converged in {iteration} iterations")
    return V