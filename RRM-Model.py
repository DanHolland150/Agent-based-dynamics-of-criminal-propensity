import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.stats import truncnorm


# Number of agents, number of time steps, and how many to plot
N = 100
T = 10000
plot_amount = 10
TIME = np.arange(0,T + 1)
A = np.arange(N)

def scaled_truncnorm(L, U, mean):
    """
     Properly scale truncnorm to the desired bounds and mean.

     Args:
        L: The desired lower bound,
        U: The desired upper bound,
        mean: The mean of the desired region.

    Returns:
        values: A vector of N values drawn from the appropriate distributions.
    """
    # Standardise the bounds for truncnorm. Note sigma = 1
    a = (L - mean)
    b = (U - mean)

    # Use truncated normal via truncnorm
    dist = truncnorm(a, b, mean)

    # Take N samples and return vector
    values = dist.rvs(N).reshape(N, 1)
    return values


# Initialize propensity in c, C will be all c values over time.
c = scaled_truncnorm(-1, 1, 0)
C = c.copy()

# Initialize r, re and p:
r = scaled_truncnorm(0, 1, 0.5)
re = scaled_truncnorm(0, 1, 0.5)
p = scaled_truncnorm(-1, 1, 0)


for t in range(1, T+1):
    # Pick pair of interacting agents (i and j):

    interactor_amount = random.randrange(2, N + 1, 2)
    interactors = np.random.permutation(np.arange(N))[:interactor_amount]
    witness_pool = np.setdiff1d(A,interactors)
    witnesses = [[] for _ in range(interactor_amount//2)]
    for witness in witness_pool:
       pair_index_seen = np.random.randint(0,interactor_amount//2)
       witnesses[pair_index_seen].append(witness)


    for interactor_index in range(0, len(interactors), 2):
       i = interactors[interactor_index]
       j = interactors[interactor_index + 1]


       # Calculate the w values that are in use for this pair. The  use of the variable s lets
       # us chose +1 or -1 with 50/50 chance:
       s = np.random.choice([-1, 1])
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
       for k in witnesses[interactor_index//2]:
         wkij = s * re[k] / 4 * (1 - abs(c[k])) * (abs(c[i] - p[k]) + abs(c[j] - p[k])) / 2
         if re[k] * (abs(c[i] - p[k]) + abs(c[j] - p[k])) / 2 > abs(p[k]):
            c[k] += wkij * (abs(c[i]-c[j]) + abs(c[i] - c[k]) + abs(c[j] - c[k])) / 3
         else:
            c[k] += (1 - abs(p[k])) * (1 - abs(c[k])) * (p[k] - c[k])

   
    # Update the c history:
    C = np.hstack((C, c))


# Plot
for f in range(plot_amount):
    plt.plot(TIME,C[f,:],linewidth=1)

plt.xlabel("Time step, t", fontsize = 14)
plt.ylabel("Propensity, c(t)", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim(0, T)
plt.tight_layout()

plt.show()