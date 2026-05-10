import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import truncnorm
# Number of agents, number of time steps, and how many to plot
N = 10
T = 5
seed = 10
TIME = np.arange(0,T + 1)
A = np.arange(N)
np.random.seed(seed)

def generate_pairs_and_witnesses():
    all_time_interactors = [[] for _ in range(T)]
    all_time_witnesses = [[] for _ in range(T)]
    pm = [[] for _ in range(T)]

    for i in range(T):
        interactor_amount = 2 + np.random.randint(N//2) * 2
        interactors = np.random.permutation(np.arange(N))[:interactor_amount]
        witness_pool = np.setdiff1d(A,interactors)
        witnesses = [[] for _ in range(interactor_amount//2)]
        for witness in witness_pool:
            pair_index_seen = np.random.randint(0,interactor_amount//2)
            witnesses[pair_index_seen].append(witness)

        all_time_interactors[i] = interactors
        all_time_witnesses[i] = witnesses
        for j in range(interactor_amount//2):
            sign = np.random.choice([-1,1])
            pm[i].append(sign)


    return all_time_interactors, all_time_witnesses, pm

all_time_interactors, all_time_witnesses, pm = generate_pairs_and_witnesses()


def scaled_truncnorm(L, U, mean, X):
    """
     Properly scale truncnorm to the desired bounds and mean.

     Args:
        L: The desired lower bound,
        U: The desired upper bound,
        mean: The mean of your desired region.

    Returns:
        values: A vector of N values drawn from the appropriate distributions.
    """
    # Standardise the bounds for truncnorm. Note sigma = 1
    a = (L - mean)
    b = (U - mean)

    # Use truncated normal via truncnorm
    dist = truncnorm(a, b, mean)

    # Take N samples and return vector
    values = dist.rvs(N, random_state = X).reshape(N, 1)
    return values

def simulate(R,P):
    r = R * np.ones(N)
    re = r
    p = P * np.ones(N)
    c = scaled_truncnorm(-1, 1, 0, X = 4)

    #X 4 before
    #Then X = 5 was used
    C = c.copy()

    for t in range(1, T+1):
        # Pick pair of interacting agents (i and j):

        for interactor_index in range(0, len(all_time_interactors[t-1]) , 2):
            i = all_time_interactors[t-1][interactor_index]
            j = all_time_interactors[t-1][interactor_index + 1]


            # Calculate the w values that are in use for this pair. The  use of the variable s lets
            # us chose +1 or -1 with 50/50 chance:
            s = pm[t - 1][interactor_index//2]
            wij = s * r[i] / 4 * (1 - abs(c[i])) * abs(c[j]-p[i])
            wji = s * r[j] / 4 * (1 - abs(c[j])) * abs(c[i]-p[j])

            #Move i (2 cases):
            if r[i] * abs(c[j]-p[i]) > abs(p[i]):
                c[i] += wij * abs(c[j]-c[i])
            else:
                c[i] += (1-abs(p[i])) * (1 - abs(c[i])) * (p[i]-c[i])

            # Move j (2 cases):
            if r[j] * abs(c[i]-p[j]) > abs(p[j]):
                c[j] += wji * abs(c[i]-c[j])
            else:
                c[j] += (1 - abs(p[j])) * (1-abs(c[j])) * (p[j]-c[j])

            ## Witnesses:
            for k in all_time_witnesses[t-1][interactor_index//2]:
                wkij = s * re[k] / 4 * (1 - abs(c[k])) * (abs(c[i] - p[k]) + abs(c[j] - p[k])) / 2
                if re[k] * (abs(c[i] - p[k]) + abs(c[j] - p[k])) / 2 > abs(p[k]):
                    c[k] += wkij * (abs(c[i]-c[j]) + abs(c[i] - c[k]) + abs(c[j] - c[k])) / 3
                else:
                    c[k] += (1 - abs(p[k])) * (1 - abs(c[k])) * (p[k] - c[k])


        # Update the c history:
        C   = np.hstack((C, c))


    if abs(np.mean(c)-P)<1e-6 and abs(np.var(c))<1e-6:
        return 0
    elif 1-abs(min(c, key = abs))<1e-6:
        return 1
    else:
        return 2
    

r_values = np.linspace(0,1,20)
p_values = np.linspace(0,0.06,20)
for u in r_values:
    for v in p_values:
        ex = simulate(u,v)
        if ex == 0:
            color = 'blue'
            plt.scatter(u,v,color = color , marker = 'o')
        elif ex == 1:
            color = 'red'
            plt.scatter(u,v,color = color, marker = ',')
        else:
            color = 'black'
            plt.scatter(u,v,color = color, marker = '^')


plt.tight_layout()
plt.xlabel('r value', fontsize = 14)
plt.ylabel('p value', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.show()