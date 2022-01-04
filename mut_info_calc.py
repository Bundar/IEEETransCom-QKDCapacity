import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy import special as sp
import math
import scipy.integrate as integrate
import csv

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def write_to_csv(ddata, name):
    f = open(f'./{name}', 'w')
    writer = csv.writer(f,delimiter = " ")
    for t, d in zip(ddata[0], ddata[1]):
        if d != 0:
            writer.writerow([t, d])
    f.close()
def gaussian_pdf(x, s):
    return 1/(math.sqrt(2*math.pi*s**2)) * math.e**(-x**2/(2*s**2))

def prob_jitter_error(j, a, b, f, s):
    p = integrate.quad(lambda x: gaussian_pdf(x, s), j*b-a, (j+1)*b-a)[0]
    return p

def avg_jitter_error_prob(j, b, f, s):
    p = 1/b * integrate.quad(lambda a: prob_jitter_error(j, a, b, f, s), 0, b)[0]
    if p > 1:
        p = 1
    return p

def get_avg_jitter_error_prob_bpf(probs, j, tau, f_size, s):
    j = int(j+(f_size/tau))
    if probs[j] > -1:
        return probs[j]
    else:
        p = avg_jitter_error_prob(j, tau, f_size, s)
        probs[j] = p
        return p


def mut_info(n, tf, s):
    support = 3
    probs = [-1 for _ in range(2*n+1)]
    dmc = [[0]*n for _ in range(n)]
    for x in range(n):
        for y in range(n):
            p = get_avg_jitter_error_prob_bpf(probs, y-x, tf/n, tf, s) #if abs(y-x) <= 3*s    /(tf/n) else 0
            dmc[x][y] = p if p > 0 else 0
        norm = sum(dmc[x])
        dmc[x] = [dmc[x][_]/norm for _ in range(n)] if norm > 0 else [0 for _ in range(n)]
    mut_info = math.log(n,2)
    for x in range(n):
        cond_entropy = 0
        for y in range(n):
            p_yx = dmc[x][y] * math.log(dmc[x][y], 2) if dmc[x][y] > 0  else 0
            cond_entropy += p_yx
        mut_info += 1/n *(cond_entropy)

    return mut_info

def iXYvsHx():
    tf = 1000
    sig_ab_1 = 50 # 0.05 sigab/tf
    sig_ab_2 = 25 # 0.025 sigab/tf
    sig_ab_3 = 12.5 # 0.0125 sigab/tf
    sig_ab_4 = 100 # 0.1 sigab/tf

    bin_counts = [i for i in range(1,16)]
    data = [[],[],[],[]]

    for i in bin_counts:
        n=2**i
        print(n)
        #data[0].append(-math.log(3*sig_ab_1/tf))
        #data[1].append(-math.log(3*sig_ab_2/tf))
        #data[2].append(-math.log(3*sig_ab_3/tf))
        data[0].append(mut_info(n, tf, sig_ab_1))
        data[1].append(mut_info(n, tf, sig_ab_2))
        data[2].append(mut_info(n, tf, sig_ab_3))
        data[3].append(mut_info(n, tf, sig_ab_4))
    plt.plot(bin_counts, data[0], label = "$\sigma_{ab}/T_f = 0.05$")
    plt.plot(bin_counts, data[1], label = "$\sigma_{ab}/T_f = 0.025$")
    plt.plot(bin_counts, data[2], label = "$\sigma_{ab}/T_f = 0.0125$")
    plt.plot(bin_counts, data[3], label = "$\sigma_{ab}/T_f = 0.1$")
    plt.xlabel("H(X)")
    plt.ylabel("I(X;Y)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    iXYvsHx()

