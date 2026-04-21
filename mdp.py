import numpy as np

STATES = ['High', 'Low', 'Charging']
ACTIONS = ['Search', 'Wait']

GAMMA = 0.9
THETA = 1e-6

def build_mdp():
    P = np.zeros((3, 2, 3))
    R = np.zeros((3, 2))

    # High
    P[0, 0, 0] = 0.7
    P[0, 0, 1] = 0.3
    R[0, 0] = 4

    P[0, 1, 0] = 1.0
    R[0, 1] = 1

    # Low
    P[1, 0, 0] = 0.4
    P[1, 0, 1] = 0.6
    R[1, 0] = -3

    P[1, 1, 1] = 1.0
    R[1, 1] = 1

    # Charging
    P[2, 1, 0] = 1.0
    R[2, 1] = 0

    return P, R