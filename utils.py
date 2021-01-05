import numpy as np
from scipy.integrate import quad

def f(x):
    return np.exp(-x)/x

def computeCapacity(a, b):
    c = a/((a - b)*np.log(2))
    if a >= 1/700 and b >= 1/700:
        int_a = quad(f, a=1/a, b=np.inf)[0]
        int_b = quad(f, a=1/b, b=np.inf)[0]
        return c * (np.exp(1/a)*int_a - np.exp(1/b)*int_b)
    elif a < 1/700 and b < 1/700:
        return c * (a - b)
    elif b < 1/700:
        int_a = quad(f, a=1 / a, b=np.inf)[0]
        return c * (np.exp(1/a)*int_a - b)
    else:
        int_b = quad(f, a=1/b, b=np.inf)[0]
        return c * (a - np.exp(1/b) * int_b)

def GetCost(costMat, assignments):
    sum_cost = 0
    min_cost = np.inf
    for row, column in assignments:
        cost = costMat[row][column]
        sum_cost += cost
        if cost < min_cost:
            min_cost = cost
    return sum_cost, min_cost

