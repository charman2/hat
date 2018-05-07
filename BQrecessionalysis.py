
import numpy as np

import pandas as pd
import matplotlib.pylab as plt
import scipy.io as spio

import RecessionAnalysis as rec
reload(rec)

mat = spio.loadmat('BoulderQ.mat', squeeze_me=True)
t = mat['BoulderQ'][()][0]
t=t-t[0]
q = mat['BoulderQ'][()][1]
correction = np.ones_like(t)
exclude = np.ones_like(t)==0
merge_gap=0
min_length=2
noise_threshold = 0.3
alpha = 0.3
epsilon = np.abs(np.diff(q)).min()
beta = 5

hydrodata = rec.Hydrograph(q, t, correction=correction)
hydrodata.getevents(exclude, noise_threshold, merge_gap, min_length)
hydrodata.getdqdt(correction=correction, alpha=alpha, beta=beta)
def func(Q, a, b):
    return a * Q**b
hydrodata.getgf(func, p0=(1,1))

plt.figure(1)
plt.clf()
axQ = plt.subplot(111)
plt.figure(2)
plt.clf()
axR = plt.subplot2grid((1,2),(0,0))
axRcts = plt.subplot2grid((1,2),(0,1), sharex=axR, sharey=axR)
axR.set_xlabel('Discharge, $Q [m^3/d]$')
axR.set_ylabel('$-dQ/dt~[m^3/d^2]$')
axR.set_xscale('log')
axR.set_yscale('log')
axR.set_title('Noise reduction method')
axRcts.set_xlabel('Discharge, $Q [m^3/d]$')
axRcts.set_ylabel('$-dQ/dt~[m^3/d^2]$')
axRcts.set_xscale('log')
axRcts.set_yscale('log')
axRcts.set_title('Constant time step method')
plt.figure(3)
plt.clf()
axg = plt.subplot2grid((1,2),(0,0))
axS = plt.subplot2grid((1,2),(0,1))
axg.set_xscale('log')
axg.set_yscale('log')
#axS.set_xscale('log')
#axS.set_yscale('log')

axQ.plot(t, q,'c.')
axQ.plot(t[~exclude], q[~exclude],'b.')
for R in hydrodata.Recessions:
    axQ.plot(R.t, R.q,'0.5', lw=2)
    axQ.plot(R.t[0], R.q[0],'g.')
    axQ.plot(R.t[-1], R.q[-1],'r.')
    axQ.plot(R.t, R.func_q(R.t-R.t[0]), 'm-', alpha=0.5, lw=1)
    for i in range(len(R.q_deriv)):
        axR.plot(R.q_deriv[i], R.dq_dt[i], 'c.', alpha=R.rating[i], markeredgecolor='none')
        axg.plot(R.q_deriv[i], R.dq_dt[i]*R.correction[i], 'b.', alpha=R.rating[i], markeredgecolor='none')
    axg.plot(R.q_deriv, R.g_func(R.q_deriv)*R.q_deriv, 'r-', alpha=0.5, lw=1)
    axRcts.plot(R.q_deriv_cts, R.dq_dt_cts, 'r.', alpha=0.3, markeredgecolor='none')
axQ.set_yscale('log')
axg.plot(hydrodata.Q, hydrodata.g_func(hydrodata.Q)*hydrodata.Q, 'b-')
axS.plot(hydrodata.S, hydrodata.Q, 'k-')
axS.set_ylabel('Discharge [mm/day]')
axS.set_xlabel('Storage [mm]')
plt.show()
