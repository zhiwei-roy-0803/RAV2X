import numpy as np
from scipy.optimize import linprog
from copy import deepcopy

class GraphMatching3D:

    def __init__(self, M, F, N):
        self.M = M
        self.F = F
        self.N = N

    def LPRelaxation(self, W):
        '''
        Linear programing provides basic solution for the integer programming
        :param W:
        :return:
        '''
        A = np.zeros((self.M+self.F+self.N, self.M*self.F*self.N))
        for i in range(self.M):
            order = 0
            for m in range(self.M):
                for f in range(self.F):
                    for n in range(self.N):
                        if m == i:
                            A[i, order] = 1
                        order += 1
        for i in range(self.F):
            order = 0
            for m in range(self.M):
                for f in range(self.F):
                    for n in range(self.N):
                        if f == i:
                            A[i+self.M, order] = 1
                        order += 1
        for i in range(self.N):
            order = 0
            for m in range(self.M):
                for f in range(self.F):
                    for n in range(self.N):
                        if n == i:
                            A[i+self.M+self.F, order] = 1
                        order += 1
        b = np.ones((self.M+self.F+self.N, 1))
        f = -np.reshape(W, newshape=(-1,))
        bounds = (0, np.inf)
        res = linprog(f, A_ub=A, b_ub=b, bounds=bounds, method="simplex")
        x = res["x"]
        fval = -res["fun"]
        X = np.reshape(x, newshape=(self.M, self.F, self.N))
        return X, fval

    def sumNeighbor(self, Y, m, f, n):
        sum0 = np.sum(Y[m, :, :])
        sum1 = np.sum(Y[:, f, :])
        sum2 = np.sum(Y[:, :, n])
        sum3 = np.sum(Y[m, f, :])
        sum4 = np.sum(Y[m, :, n])
        sum5 = np.sum(Y[:, f, n])
        sum6 = Y[m, n, f]
        return sum0 + sum1 + sum2 - sum3 - sum4 -sum5 + sum6

    def Judge(self, result1, result2):
        long = result1.shape[0]
        isMatch = True
        for i in range(2, long, 3):
            if result2[0] == result1[i - 2]:
                isMatch = False
                break
            if result2[1] == result1[i - 1]:
                isMatch = False
                break
            if result2[2] == result1[i]:
                isMatch = False
                break
        return isMatch


    def localRatio(self, F, W, sequence):
        # Implement Local Ratio Method
        wM, wF, wN = F.shape
        sz = len(sequence)
        for i in range(2, sz, 3):
            m = sequence[i - 2]
            f = sequence[i - 1]
            n = sequence[i]
            if F[m, f, n] != 0:
                value = W[m, f, n]
                check = np.zeros((wM, wF, wN))
                # for each m
                for a in range(wF):
                    for b in range(wN):
                        if check[m, a, b] == 0:
                            check[m, a, b] = 1
                            if W[m, a, b] > value:
                                W[m, a, b] -= value
                            else:
                                W[m, a, b] = 0
                                F[m, a, b] = 0
                # for each f
                for a in range(wM):
                    for b in range(wN):
                        if check[a, f, b] == 0:
                            check[a, f, b] = 1
                            if W[a, f, b] > value:
                                W[a, f, b] -= value
                            else:
                                W[a, f, b] = 0
                                F[a, f, b] = 0
                # for each n
                for a in range(wM):
                    for b in range(wF):
                        if check[a, b, n] == 0:
                            check[a, b, n] = 1
                            if W[a, b, n] > value:
                                W[a, b, n] -= value
                            else:
                                W[a, b, n] = 0
                                F[a, b, n] = 0
                # Now, we judge whether or not we need to continue to next LocalRatio
                KMatrix = F != np.zeros((wM, wF, wN))
                sumKMatrix = np.sum(KMatrix)
                if sumKMatrix > 0:
                    sequence = sequence[i+1:]
                    result2 = np.array([m, f, n])
                    result1 = self.localRatio(F, W, sequence)
                    if self.Judge(result1, result2) == True:
                        szResult = result1.shape[0]
                        result = np.zeros(szResult+3, dtype=np.int)
                        result[:3] = result2
                        result[3:] = result1
                        return result
                    else:
                        return result1
                else:
                    return np.array([m, f, n])



    def weighted_D3Matching(self, X, W):
        inside = np.ones((self.M, self.F, self.N), dtype=np.int)
        F = np.zeros_like(inside)
        sequence = np.zeros(3*self.M*self.F*self.N, dtype=np.int)
        order = 0
        # Prepare for F matrix in Algorithm5
        while order < self.M*self.F*self.N:
            Y = X * inside
            for m in range(self.M):
                for f in range(self.F):
                    for n in range(self.N):
                        if inside[m, f, n] == 1:
                            sumValue = self.sumNeighbor(Y, m, f, n)
                            if sumValue <= 2:
                                F[m, f, n] = order
                                sequence[3*order] = m
                                sequence[3*order+1] = f
                                sequence[3*order+2] = n
                                order += 1
                                inside[m, f, n] = 0
                                Y[m, f, n] = 0
        F *= (W != 0)
        # Local Ratio Algorithm (Algorithm 6)
        W_copy = deepcopy(W) # localRatio algorithm will modify W matrix, therefore we should pass its copy as input
        result = self.localRatio(F, W_copy, sequence)
        # Greedy algorithm to find a maximal set (step 10 in Algorithm 5)
        yui1 = np.zeros(self.M)
        yui2 = np.zeros(self.F)
        yui3 = np.zeros(self.N)
        Z = np.zeros((self.M, self.F, self.N))
        for i in range(2, len(result), 3):
            vertex1 = result[i - 2]
            vertex2 = result[i - 1]
            vertex3 = result[i]
            yui1[vertex1] = 1
            yui2[vertex2] = 1
            yui3[vertex3] = 1
            Z[vertex1, vertex2, vertex3] = 1
        for m in range(self.M):
            for f in range(self.F):
                for n in range(self.N):
                    if yui1[m] == 0 and yui2[f] == 0 and yui3[n] == 0:
                        if W[m, f, n] > 0:
                            Z[m, f, n] = 1
                            yui1[m] = 1
                            yui2[f] = 1
                            yui3[n] = 1
        # Algorithm 5 finished, obtain the approximate results for the weighted 3-Matching Problem
        return Z, yui1, yui2, yui3, np.sum(Z * W)


    def run_3DMatching(self, C):
        X, rateLP = self.LPRelaxation(C)
        Z, vertex1, vertex2, vertex3, fval = self.weighted_D3Matching(X, C)
        return Z, vertex1, vertex2, vertex3, fval
