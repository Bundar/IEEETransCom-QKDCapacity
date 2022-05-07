import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy import special as sp
import math
import scipy.integrate as integrate
import csv
from scipy import special

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def write_to_csv(ddata, name):
    print(name)
    f = open('./'+name, 'w')
    writer = csv.writer(f,delimiter = ",")
    for t, d in zip(ddata[0], ddata[1]):
        if d != 0:
            writer.writerow([t, d])
    f.close()

def c(f, lp, ldc):
    return 1
    tf = 330
    pppmspdc = lp*tf*math.e**(-lp*tf)*(math.e**(-ldc*tf))**2
    pppmdc = math.e**(-lp*tf)*(ldc*tf*math.e**(-ldc*tf))**2
    ptot = pppmspdc+pppmdc
    return pppmspdc/ptot if ptot > 0 else oops(f,lp,ldc,pppmspdc,pppmdc,ptot)

def ptotal(f,lp,ldc):
    tf = 330
    pppmspdc = lp*tf*math.e**(-lp*tf)*(math.e**(-ldc*tf))**2
    pppmdc = math.e**(-lp*tf)*(ldc*tf*math.e**(-ldc*tf))**2
    ptot = pppmspdc+pppmdc
    return ptot

def oops(f,lp,ldc,pppmspdc,pppmdc,ptot):

#    print(f"oops: f: {f},lp: {lp},ldc: {ldc},pppmspdc: {pppmspdc},pppmdc: {pppmdc},ptot:{ptot}")
    return 0

def qfunc(x, s):
    return 0.5 - 0.5*special.erf((x/s)/np.sqrt(2))

def gauss_integrated(A,B,s):
    return qfunc(A,s)-qfunc(B,s)

def tri_integrated(A,B,f):
    return 0.5*(B-A)*(tri(f,A)+tri(f,B))
def tri(f,x):
    return 1/f + abs(x)/(f**2)

def prob_jitter_error(j, a, b, f, s, lp, ldc):
    A = j*b-a
    B = j*b+b-a
    c_val = c(f,lp,ldc)
    perr = c_val*gauss_integrated(A,B,s) + (1-c_val)*tri_integrated(A,B,f)
    return perr

def avg_jitter_error_prob(j, b, f, s, lp, ldc):
    p = 1/b * integrate.quad(lambda a: prob_jitter_error(j, a, b, f, s, lp, ldc), 0, b)[0]
    if p > 1:
        p = 1
    return p

def h(p):
    if p <= 0:
        return 0
    return p*math.log(p,2)

def mut_info(n, tf, s, lp, ldc):
    support = 3*s
    ixy = math.log(n)
    hyx = 0
    for j in range(-(n-1), n-1):
        hyx += (n-abs(j))*h(avg_jitter_error_prob(j, tf/n, tf, s, lp, ldc))
    ixy += hyx/n

    return ixy

def error_rate(n, tf, s):
    ser = 1 - avg_jitter_error_prob(0, tf/n, tf, s,0 ,0)
    return ser 

def frameErrorRateVsHx():
    graphtitle = "SymbolErrorRateVsBinCount"
    tf = 1000
    sig_ab_2 = 0.102948#fwhm jitter 80ps, 330ns Tf
    sig_ab_1 = 0.5*sig_ab_2
    sig_ab_3 = 2*sig_ab_2

    bit_counts = [i for i in range(1,18)]
    data = [[],[],[]]

    for i in bit_counts:
        n=2**i
        print(n)
        data[0].append(error_rate(n, tf, sig_ab_1))
        data[1].append(error_rate(n, tf, sig_ab_2))
        data[2].append(error_rate(n, tf, sig_ab_3))
    plt.plot(bit_counts, data[0], label = "$\sigma_{ab}/T_f ="+str(sig_ab_1/tf)+"$")
    plt.plot(bit_counts, data[1], label = "$\sigma_{ab}/T_f ="+str(sig_ab_2/tf)+"$")
    plt.plot(bit_counts, data[2], label = "$\sigma_{ab}/T_f ="+str(sig_ab_3/tf)+"$")
    plt.xlabel("H(X)")
    plt.ylabel("Symbol Error Rate")
    
    write_to_csv([bit_counts,data[0]] ,"./datafiles/"+graphtitle+"_s"+str(sig_ab_1/tf)+".csv")
    write_to_csv([bit_counts, data[1]], "./datafiles/"+graphtitle+"_s"+str(sig_ab_2/tf)+".csv")
    write_to_csv([bit_counts, data[2]], "./datafiles/"+graphtitle+"_s"+str(sig_ab_3/tf)+".csv")

    plt.legend()
    
    plt.savefig("./plots/"+graphtitle)

def maxIXYvssigmaRat():
    graphtitle = "MaxMutualInfoVsSigRat" 
    tf = 1000 
    sigrats = [i/1000/tf for i in range(1, 20000, 500)]
    bin_counts = [i for i in range(1,18)]
    maxIXYs = []
    maxAchievingBinRat = []
    for sigrat in sigrats:
        print(f"sigrat: {sigrat}")
        tempmax = 0
        binRat = None
        for i in bin_counts:
            n=2**i
            print(f"\t{n}")
            ixy = mut_info(n, tf, sigrat)
            if ixy > tempmax:
                tempmax = ixy
                binRat = sigrat*tf/(tf/n) #i#sigrat*tf/(tf/n)
            if ixy < tempmax:
                break
        maxIXYs.append(ixy)
        maxAchievingBinRat.append(binRat)
    
    plt.plot(sigrats, maxIXYs)
    plt.xlabel("Ratio of Variance to Frame Width")
    plt.ylabel("Max I(X;Y)")
    plt.savefig("./plots/"+graphtitle)
    plt.clf() 
    write_to_csv([sigrats, maxIXYs], "./datafiles/"+graphtitle+".csv")
    write_to_csv([sigrats, maxAchievingBinRat], "./datafiles/tauoversigmacomp.csv")

    plt.plot(sigrats, maxAchievingBinRat)
    plt.xlabel("Ratio of Variance to Frame Width")
    plt.ylabel("sigma/tau achieving Max I(X;Y)")
    plt.savefig("./plots/tauoversigmacomp")

def iXYvsHx():
    graphtitle = "MutualInfoVsBinCount" 
    tf = 1000
    sig_ab_2 = 0.102948#fwhm jitter 80ps, 330ns Tf
    sig_ab_1 = 0.5*sig_ab_2
    sig_ab_3 = 2*sig_ab_2

    bin_counts = [i for i in range(1,18)]
    data = [[],[],[]]
    for i in bin_counts:
        n=2**i
        print(n)
        data[0].append(mut_info(n, tf, sig_ab_1,0,0))
        data[1].append(mut_info(n, tf, sig_ab_2,0,0))
        data[2].append(mut_info(n, tf, sig_ab_3,0,0))
    plt.plot(bin_counts, data[0], label = "$\sigma_{ab}/T_f ="+str(sig_ab_1/1000)+"$")
    plt.plot(bin_counts, data[1], label = "$\sigma_{ab}/T_f ="+str(sig_ab_2/1000)+"$")
    plt.plot(bin_counts, data[2], label = "$\sigma_{ab}/T_f ="+str(sig_ab_3/1000)+"$")
    plt.plot(bin_counts, bin_counts, label = "$\sigma_{ab}/T_f = 0$")
    plt.xlabel("H(X)")
    plt.ylabel("I(X;Y)")
    write_to_csv([bin_counts, data[0]], "./datafiles/"+graphtitle+"_s"+str(sig_ab_1/1000)+".csv")
    write_to_csv([bin_counts, data[1]], "./datafiles/"+graphtitle+"_s"+str(sig_ab_2/1000)+".csv")
    write_to_csv([bin_counts, data[2]], "./datafiles/"+graphtitle+"_s"+str(sig_ab_3/1000)+".csv")
    plt.legend()
    plt.savefig("./plots/"+graphtitle)

def hXtoptau():
    tf = 1000
    hxs = [i/10 for i in range(1,180)]
    data = {}
    for h in hxs:
        n = 2**h
        print(n)
        lambdatfs = [10**i for i in range(1, 5)]
        for lambtf in lambdatfs:
            pt = (lambtf/2**h) *math.e**(-lambtf/2**h)
            if lambtf in data:
                data[lambtf].append(pt)
            else:
                data[lambtf] = [pt]

    for ltf in data:
        d = data[ltf]
        plt.plot(hxs, d, label="$\lambda_p*T_f = "+str(ltf)+"$")
        write_to_csv([hxs, d], "./datafiles/ptauvshx_"+str(ltf)+".csv")

    plt.legend()
    plt.xlabel("H(X)")
    plt.ylabel("P tau")
    plt.savefig("./plots/ptauvshx.png")

def perr():
    tf = 330
    lp = 1/tf
    ldcs = [p/50000 * lp for p in range(0, 1000)]
    prob_valids = []
    pppmspdc = lp*tf*math.e**(-lp*tf)

    for ldc in ldcs:
        pdc = math.e**(-ldc*tf)
        pvalid = pppmspdc*(pdc)**2 + (math.e**(-lp*tf))*(ldc*tf*math.e**(-ldc*tf))**2 
        prob_valids.append(pvalid)
    xs = [ldc/lp for ldc in ldcs]
    plt.plot(xs, prob_valids)
    plt.xlabel("$\lambda_{dc}/\lambda_{p}$")
    plt.title("Prob. of Observing PPM Valid Frame")
    plt.savefig("./plots/probppmvalid.png")
    write_to_csv([xs, prob_valids], "probppmvalid.csv")

def mutinfowithdcs():
    tf = 1000
    s2 = 0.102948#fwhm jitter 80ps, 330ns Tf
    s1 = 0.5*s2
    s3 = 2*s2

    lp = 1/tf
    x_range = [p/150 * lp for p in range(0, 100)]
    n = 2**12
    plot_functions = [lambda ldc: mut_info(n, tf, s1, lp, ldc)*ptotal(tf, lp,ldc)/tf*10**3, lambda ldc: mut_info(n, tf, s2, lp, ldc)*ptotal(tf, lp,ldc)/tf*10**3, lambda ldc: mut_info(n, tf, s3, lp, ldc)*ptotal(tf, lp,ldc)/tf*10**3]
    x_label = "$\lambda_{dc}/\lambda_p$"
    y_label = "Post Reconcilliation key Rate (MHZ)"
    title = "PRRateBitspTime"#"ixy_vs_lambdadc_rat_HX12_wider_range"
    labels = ["$\sigma_d/T_f = {s1:.6f}$", f"$\sigma_d/T_f = {s2:.6f}$", f"$\sigma_d/T_f = {s3:.6f}$"]
    plot_data(plot_functions, x_range, [], labels, x_label, y_label, title)

def test():
    x_range = [_ for _ in range(100)]
    plot_functions = [lambda x: x**2, lambda y: 2*y**2]
    inputs = []
    x_label = "x label here"
    y_label = "y label here"
    title = "this is the title"
    labels = ["$x^2$", "$2x^2$"]
    plot_data(plot_functions, x_range, inputs, labels, x_label, y_label, title)
    return None
    #TODO: assume symmetrical.. test symmetry of DMC... 
    n = 8
    tf = 1000
    s = 0.192333
    lp = 1
    ldc = 0.00002
    dmc = [[avg_jitter_error_prob(y-x,tf/n, tf, s, lp, ldc) for x in range(n)] for y in range(n)]
    print(dmc)
    for i in range(n):
        print(f'sum of row = {sum(dmc[i])}')

    ixy = math.log(n)
    hyx = 0
    for x in range(n):
        for y in range(n):
            hyx += h(dmc[y][x])
    ixy += hyx / n
    print(f'ixy full dmc: {ixy}')

    for x in range(n):
        ixy = math.log(n)
        hxy = 0
        for y in range(n):
            hxy += h(dmc[y][x])

        print(f'row {x} gives avg ixy: {ixy - hxy}')

def plot_data(plot_functions, x_range, inputs,labels, x_label, y_label, title):
    data= []
    for f,func in enumerate(plot_functions):
        f_data = []
        print(f"Calculating Plot {f}:\n")
        for i,x in enumerate(x_range):
            j = (i+1)/len(x_range)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            f_data.append(func(x))
        data.append(f_data)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    for i, ys in enumerate(data):
        plt.plot(x_range, ys, label=labels[i])
        write_to_csv([x_range, ys], "./datafiles/"+title+str(i)+".csv")

    plt.savefig(f"./plots/{title}.png")
    return None

if __name__ == "__main__":
    #maxIXYvssigmaRat()
    #iXYvsHx()
    #frameErrorRateVsHx()
    #hXtoptvau()
    #perr()
    mutinfowithdcs()
    #test()
