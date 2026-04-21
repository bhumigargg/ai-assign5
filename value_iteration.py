import numpy as np

def value_iteration(P, R, gamma, theta):
    V = np.zeros(len(P))
    history = []

    while True:
        delta = 0
        V_new = np.copy(V)

        for s in range(len(P)):
            Q = []
            for a in range(len(P[s])):
                q = sum(P[s, a, s_next] *
                        (R[s, a] + gamma * V[s_next])
                        for s_next in range(len(P)))
                Q.append(q)

            V_new[s] = max(Q)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        history.append(V.copy())

        if delta < theta:
            break

    return V, history


def extract_policy(V, P, R, gamma):
    policy = []

    for s in range(len(P)):
        Q = []
        for a in range(len(P[s])):
            q = sum(P[s, a, s_next] *
                    (R[s, a] + gamma * V[s_next])
                    for s_next in range(len(P)))
            Q.append(q)

        policy.append(int(np.argmax(Q)))

    return policy