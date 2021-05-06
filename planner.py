#! /usr/bin/python3

'''
CS 747: Programming Assignment 2
Mohd Safwan, October 2020
'''

import numpy as np
import pulp as pl
import argparse


class MDP:

    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.n = int(f.readline().split()[-1])
            self.k = int(f.readline().split()[-1])
            self.start = int(f.readline().split()[-1])
            self.end = [int(i) for i in f.readline().split()[1:]]
            self.R = np.zeros((self.n, self.k, self.n))
            self.T = np.zeros((self.n, self.k, self.n))
            while True:
                data = f.readline().split()
                if len(data) <= 2:
                    break
                i = int(data[1])
                j = int(data[2])
                k = int(data[3])
                self.R[i, j, k] = float(data[4])
                self.T[i, j, k] = float(data[5])
            self.continuing = (data[-1] == 'continuing')
            data = f.readline().split()
            self.gamma = float(data[1])

    def policy_evaluation(self, pi, b):
        b = np.take_along_axis(b, pi[:, None], 1)
        T = np.take_along_axis(self.T, pi[:, None, None], 1)[:, 0]
        A = np.eye(self.n) - self.gamma * T
        try:
            V = np.linalg.solve(A, b).ravel()
        except:
            V = np.linalg.lstsq(A, b, rcond=None)[0].ravel()
        return V

    def solve_HPI(self):
        A = (self.T * self.R).sum(2)
        b = A.copy()
        B = self.gamma * self.T
        pi = np.random.randint(self.k, size=self.n)
        while True:
            V = self.policy_evaluation(pi, b)
            Q = A + (B * V[None, None, :]).sum(2)
            pi_ = np.argmax(Q, 1)
            if np.all(pi == pi_):
                break
            pi = pi_
        return V, pi

    def solve_VI(self):
        epsilon = 1e-4
        dtype = np.float
        threshold = 1e-12
        if self.gamma < 1:
            dtype = np.float32
            threshold = epsilon*(1/self.gamma - 1)
            self.T = self.T.astype(dtype)
            self.R = self.R.astype(dtype)
        A = (self.T * self.R).sum(2)
        B = self.gamma * self.T
        V = np.zeros(self.n, dtype)
        while True:
            Q_n = A + (B * V[None, None, :]).sum(2)
            V_n = Q_n.max(1)
            if (V_n - V).max() < threshold:
                break
            V = V_n
        pi_star = np.argmax(Q_n, 1)
        return V_n, pi_star

    def solve_LP(self):
        mdp_prob = pl.LpProblem("mdp", pl.LpMinimize)
        V = [pl.LpVariable(f'V_{i}') for i in range(self.n)]
        mdp_prob += pl.lpSum(V)
        B = (self.T * self.R).sum(2)
        for j in range(self.k):
            t = self.T[:, j]*self.gamma
            b = B[:, j]
            for i in range(self.n):
                A = t[i]
                A[i] -= 1
                mdp_prob += pl.lpSum(V*A) + b[i] <= 0
        mdp_prob.solve(pl.PULP_CBC_CMD(msg=0))
        V_star = np.array([pl.value(v) for v in V])
        Q_star = B + (self.T * (self.gamma * V_star)[None, None, :]).sum(2)
        pi_star = np.argmax(Q_star, 1)
        return V_star, pi_star


def driver(mdp, algorithm):
    np.random.seed(42)
    m = MDP(mdp)
    solvers = {
        'lp': m.solve_LP,
        'vi': m.solve_VI,
        'hpi': m.solve_HPI
    }
    V, pi = solvers[algorithm]()
    out = ''
    for i in range(m.n):
        out += f"{V[i]:.06f}\t{pi[i]}\n"
    out = out[:-1]
    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', default='data/mdp/continuing-mdp-10-5.txt')
    parser.add_argument('--algorithm', default='vi',
                        choices=['lp', 'vi', 'hpi'],)
    args = parser.parse_args()
    out = driver(args.mdp, args.algorithm)
    print(out)
