import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.stats import truncnorm



# Number of agents, number of time steps, and how many to plot
N = 100
TIME = 10000
plot_amount = 10
T = np.arange(1,TIME + 1)
A = np.arange(N)

def scaled_truncnorm(L, U, mean):
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
    values = dist.rvs(N).reshape(N, 1)
    return values

# Initialize propensity in c, C will be all c values over time.
c = scaled_truncnorm(-1, 1, 0)
C = c.copy()

# Initialize r, re and p:
r = scaled_truncnorm(0, 1, 0.5)
re = scaled_truncnorm(0, 1, 0.5)
p = scaled_truncnorm(-1, 1, 0)

for t in range(1, TIME):
    # Pick pair of interacting agents (i and j):
    involved_amount = random.randint(2,N)
    involved = np.random.permutation(np.arange(N))[:involved_amount]
    i = involved[0]
    j = involved[1]
    Witnesses = np.setdiff1d(involved,[i,j])
    M = np.setdiff1d(A,involved)

    # Calculate the w values that are in use for this pair. The  use of the variable s lets
    #  us chose +1 or -1 with 50/50 chance:
    s = np.random.choice([-1, 1])
    wij = s * r[i] / 4 * (1 - abs(c[i])) * abs(c[j]-p[i])
    #print(r[i])
    wji = s * r[j] / 4 * (1 - abs(c[j])) * abs(c[i]-p[j])

    # Move i (2 cases):
    if r[i] * abs(c[j]-p[i]) > abs(p[i]):
        c[i] += wij * abs(c[j]-c[i])
    else:
        c[i] += (1-abs(p[i])) * (1 - abs(c[i])) * (p[i]-c[i])


    # Move j (2 cases):
    if r[j] * abs(c[i]-p[j]) > abs(p[j]):
        c[j] += wji * abs(c[i]-c[j])
    else:
        c[j] += (1 - abs(p[j])) * (1-abs(c[j])) * (p[j]-c[j])


    # Witnesses:
    for k in Witnesses:
      wkij = s * re[k] / 4 * (1 - abs(c[k])) * (abs(c[i] - p[k]) + abs(c[j] - p[k])) / 2

      if re[k] * (abs(c[i] - p[k]) + abs(c[j] - p[k])) / 2 > abs(p[k]):
        c[k] += wkij * (abs(c[i]-c[j]) + abs(c[i] - c[k]) + abs(c[j] - c[k])) / 3
      else:
        c[k] += (1 - abs(p[k])) * (1 - abs(c[k])) * (p[k] - c[k])

    # The left over agents
    for m in M:
      c[m] += (1 - abs(p[m])) * (1-abs(c[m])) * (p[m]-c[m])


    # Update the c history:
    C = np.hstack((C, c))


# Plot
for f in range(plot_amount):
    plt.plot(T,C[f,:],linewidth=1)

plt.xlabel("Time step")
plt.ylabel("Propensity c")
plt.xlim(1, TIME)
plt.tight_layout()
plt.show()