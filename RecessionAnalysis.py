from __future__ import print_function
import numpy as np 
import numpy.linalg
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

_verbose = True

class Hydrograph(object):
    def __init__(self, q, t=None, correction=None, epsilon=None):
        self.q = q
        if t is None:
            self.t = np.ones_like(self.q)
        else:
            self.t = t
        if epsilon is None:
            self.epsilon = np.abs(np.diff(np.unique(np.sort(self.q)))).min()
        else:
            self.epsilon = epsilon
        if correction is None:
            self.correction = np.ones_like(self.q)
        else:
            self.correction = correction

    def getevents(self, exclude=None, noise_threshold=0.05, merge_gap=0, min_length=4):
        if exclude is None:
            exclude = np.ones_like(self.t)==0
        # create a timeseries that is True if the discharge is declining
        Dq = np.diff(self.q)/self.q[:-1]
        dec = np.logical_and(Dq<0, ~exclude[:-1])
        dec = np.r_[dec, False]
        # find the start and end of the recession events
        i_start = np.where(np.logical_and(dec[1:], ~dec[:-1]))[0]+1
        if dec[0]:
            i_start = np.r_[0, i_start]
        i_end = np.where(np.logical_and(dec[:-1], ~dec[1:]))[0]+1
        if dec[-1]:
            i_end = np.r_[i_end]
        # merge events if they are mergable
        if merge_gap>0:
            for i in range(len(i_start)-2, -1, -1):
                gap = i_start[i+1] - i_end[i]
                any_exclude = np.any(exclude[i_end[i]:i_start[i+1]+1])
                jump = (self.q[i_start[i+1]] - self.q[i_end[i]])/self.q[i_end[i]]
                printv(i, gap, jump, )
                if ((gap<=merge_gap) & (jump<=noise_threshold)):
                    printv('merged')
                    i_end = np.delete(i_end, i)
                    i_start = np.delete(i_start, i+1)
                else:
                    printv('')
        # remove if they are too short
        if min_length>1:
            too_short = (i_end - i_start)<min_length
            i_end = i_end[~too_short]
            i_start = i_start[~too_short]
        # return the start and end of the events
        self.Recessions = [Recession(i_start[i] , i_end[i] , self.q[i_start[i]:i_end[i]+1] , self.t[i_start[i]:i_end[i]+1]) for i in range(len(i_start))]

    def getdqdt(self, **kwargs):
        if 'correction' in kwargs:
            self.correction = kwargs['correction']
        self.t_deriv = np.array([])
        self.q_deriv = np.array([])
        self.dq_dt = np.array([])
        self.correction_deriv = np.array([])
        self.rating = np.array([])
        self.t_deriv_cts = np.array([])
        self.q_deriv_cts = np.array([])
        self.dq_dt_cts = np.array([])
        for R in self.Recessions:
            R.correction = self.correction[R.i_start:R.i_end+1]
            R.getdqdt_expfit(epsilon=self.epsilon, **kwargs)
            R.getdqdt_cts()
            # Collate timeseries
            self.t_deriv = np.append(self.t_deriv, R.t_deriv)
            self.q_deriv = np.append(self.q_deriv, R.q_deriv)
            self.dq_dt = np.append(self.dq_dt, R.dq_dt)
            self.correction_deriv = np.append(self.correction_deriv, R.correction_deriv)
            self.rating = np.append(self.rating, R.rating)
            self.t_deriv_cts = np.append(self.t_deriv_cts, R.t_deriv_cts)
            self.q_deriv_cts = np.append(self.q_deriv_cts, R.q_deriv_cts)
            self.dq_dt_cts = np.append(self.dq_dt_cts, R.dq_dt_cts)

    def _fitgfunc(self, obj, g_func, p0=None):
        x = np.log(obj.q_deriv)
        y = np.log(obj.dq_dt * obj.correction_deriv)
        def func(x, *args):
            Q = np.exp(x)
            return np.log(g_func(Q, *args)*Q)
        popt, pcov = curve_fit(func, x, y, sigma=1./obj.rating, absolute_sigma=False, p0=p0)
        return lambda x: g_func(x, *popt), popt

    def getgf(self, g_func, p0=None):
        self.g_func, self.g_func_params = self._fitgfunc(self, g_func, p0)
        for R in self.Recessions:
            try:
                R.g_func, R.g_func_params = self._fitgfunc(R, g_func, p0)
            except:
                R.g_func, R.g_func_params = self.g_func, self.g_func_params
        Q = np.exp(np.linspace(np.log(self.q_deriv.min()), np.log(self.q_deriv.max()), 1000))
        g = self.g_func(Q)
        self.S = cumtrapz(1/g, Q, initial=0)
        self.Q = Q
        self.g = interp1d(Q,g)
        self.f = interp1d(Q,self.S)

class Recession(object):
    def __init__(self, i_start, i_end, q, t):
        self.i_start = i_start
        self.i_end = i_end
        self.q = q
        self.t = t

    def getdqdt_expfit(self, alpha=0.2, beta=3, epsilon=None, **kwargs):
        # Handle the inputs
        if epsilon is None:
            epsilon = np.abs(np.diff(self.q)).min()
        betaepsilon = beta * epsilon
        printv('q', self.q)
        printv('t', self.t)
        printv('alpha = ', alpha)
        printv('betaepsilon = ', betaepsilon)
        #
        # Fit exponential to the recession
        # First guess the parameters
        Qb = self.q[-1]
        dQ = np.abs(self.q[0] - Qb)
        k = 1./(self.t[-1]-self.t[0])
        printv('guess: ', Qb, k, dQ)
        def func_q(t, Qb, k, dQ):
            return Qb + dQ * np.exp(-k * t)
        popt, pcov = curve_fit(func_q, self.t-self.t[0], self.q, p0=(Qb, k, dQ), bounds=([0]*3, [np.inf]*3))
        self.Qb, self.k, self.dQ = popt
        self.func_q = lambda tt: func_q(tt, *popt)
        printv('fit: ', self.Qb, self.k, self.dQ)
        #
        # Calculate the appropriate interval over which to take a derivative
        m = np.log((self.q * (1+alpha) + betaepsilon-Qb)/(self.q - Qb)) / k
        m = np.maximum(1, np.ceil(m).astype(np.int))
        printv('m', m)
        #
        # Calculate the derivatives
        i_start_deriv = np.arange(len(self.t))-m
        self.dq_dt = np.zeros(len(self.t))
        self.q_deriv = np.zeros(len(self.t))
        self.t_deriv = np.zeros(len(self.t))
        self.Rsq = np.zeros(len(self.t))
        self.correction_deriv = np.zeros(len(self.t))
        self.qual = np.zeros(len(self.t))
        self.deriv_fit_x = [0] * len(self.t)
        self.deriv_fit_y = [0] * len(self.t)
        self.deriv_fit_ypred = [0] * len(self.t)
        for j in range(1, len(self.t)):
            if i_start_deriv[j]<0:
                self.qual[j] = j*1./m[j]
                i_start_deriv[j] = 0
            else:
                self.qual[j] = 1.
            if self.qual[j]>0:
                x=self.t[i_start_deriv[j]:j+1]
                X=np.c_[np.ones(len(x)), x]
                Y=self.q[i_start_deriv[j]:j+1]
                popt_l, _, _, _ = numpy.linalg.lstsq(X,Y, rcond=None)
                ypred = np.dot(X,popt_l)
                self.deriv_fit_x[j] = x
                self.deriv_fit_y[j] = Y
                self.deriv_fit_ypred[j] = ypred
                self.Rsq[j] = 1 - np.sum((Y - ypred)**2)/np.sum((Y - np.mean(Y))**2)
                self.Rsq[self.Rsq<0]=0
                self.dq_dt[j] = -1*popt_l[1]
                self.q_deriv[j] = np.nanmean(Y)
                self.t_deriv[j] = np.nanmean(x)
                self.correction_deriv[j] = np.nanmean(self.correction[i_start_deriv[j]:j+1])
        #
        # remove derivatives with the wrong sign
        discard = self.dq_dt<=0
        printv('discard', discard)
        self.dq_dt = self.dq_dt[~discard]
        self.t_deriv = self.t_deriv[~discard]
        self.correction_deriv = self.correction_deriv[~discard]
        self.q_deriv = self.q_deriv[~discard]
        self.Rsq = self.Rsq[~discard]
        self.qual = self.qual[~discard]
        self.rating = self.Rsq*self.qual
        printv('q_deriv = ', self.q_deriv)
        printv('self.dq_dt = ', self.dq_dt)
        printv('sigma=1./Rsq*qual = ', 1./self.Rsq*self.qual)

    def getdqdt_cts(self):
        self.dq_dt_cts = -np.diff(self.q)/np.diff(self.t)
        self.q_deriv_cts = (self.q[:-1]+self.q[1:])/2
        self.t_deriv_cts = (self.t[:-1]+self.t[1:])/2

    def getdqdt_powfit(self, alpha=0.2, beta=3, epsilon=None):
        # THIS IS CURRENTLY NOT USED
        # Handle the inputs
        if epsilon is None:
            epsilon = np.abs(np.diff(self.q)).min()
        betaepsilon = beta * epsilon
        printv('alpha = ', alpha)
        printv('betaepsilon = ', betaepsilon)
        # Loop over the recessions
        printv('q', self.q)
        printv('t', self.t)
        # Fit power function to the recession
        # First guess the parameters
        b = 2.
        Qr = self.q.min()/self.q.max()
        oQr = 1-1/(1-(self.q[-1]/self.q[0])**(1-b))
        tau = -(1-b)*oQr*(self.t[-1]-self.t[0])
        Q0 = self.q[-1]
        aguess = Q0**(1-b)/tau
        bguess = b
        printv('guess: ', Q0, tau, b)
        printv('guess: ', aguess, bguess)
        # Try fitting a power function. If that fails try an exponential
        #try:
        if True:
            def func_t(t, Q0, tau, b):
                return Q0 * (1-(1-b) * t/tau)**(1./(1-b))
            popt, pcov = curve_fit(func_t, self.t-self.t[-1], self.q, p0=(Q0, tau, b))
            Q0, tau, b = popt
            afit = Q0**(1-b)/tau
            bfit = b
        #except:
        #    def func_t(t, Q0, tau):
        #        return Q0 * np.exp(-t/tau)
        #    popt, pcov = curve_fit(func_t, self.t-self.t[-1], self.q, p0=(Q0, tau))
        #    Q0, tau = popt
        #    b = 1.
        #    afit = 1/tau
        #    bfit = 1.
        printv('fit:   ', Q0, tau, b)
        printv('fit:   ', afit, bfit)
        #if axQ is not None:
            #axQ.plot(self.t, func_t(self.t-self.t[-1], *popt),'m-')
    #    if self is not None:
    #        self.plot(self.q,afit*self.q**bfit,'m--', alpha=0.4)
        # Calculate the appropriate interval over which to take a derivative
        m = ((self.q/Q0)**(1-b) - ((self.q * (1+alpha) + betaepsilon)/Q0)**(1-b)) * tau / (b-1)
        m = np.ceil(m).astype(np.int)
        printv('m', m)
        #
        i_start_deriv = np.arange(len(self.t))-m
        dq_dt = np.zeros(len(self.t))
        q_deriv = np.zeros(len(self.t))
        t_deriv = np.zeros(len(self.t))
        Rsq = np.zeros(len(self.t))
        correction_deriv = np.zeros(len(self.t))
        qual = np.zeros(len(self.t))
        deriv_fit_x = [0] * len(self.t)
        deriv_fit_y = [0] * len(self.t)
        deriv_fit_ypred = [0] * len(self.t)
        #
        for j in range(1, len(self.t)):
            if i_start_deriv[j]<0:
                qual[j] = j*1./m[j]
                i_start_deriv[j] = 0
            else:
                qual[j] = 1.
            if qual[j]>0:
                x=self.t[i_start_deriv[j]:j+1]
                X=np.c_[np.ones(len(x)), x]
                Y=self.q[i_start_deriv[j]:j+1]
                popt, _, _, _ = numpy.linalg.lstsq(X,Y, rcond=None)
                ypred = np.dot(X,popt)
                deriv_fit_x[j] = x
                deriv_fit_y[j] = Y
                deriv_fit_ypred[j] = ypred
                Rsq[j] = 1 - np.sum((Y - ypred)**2)/np.sum((Y - np.mean(Y))**2)
                Rsq[Rsq<0]=0
                dq_dt[j] = -1*popt[1]
                q_deriv[j] = np.nanmean(Y)
                t_deriv[j] = np.nanmean(x)
                correction_deriv[j] = np.nanmean(self.correction[i_start_deriv[j]:j+1])
                #if axQ is not None:
                    #axQ.plot(x, ypred,'g-', lw=0.5)
                #if axR is not None:
                    #axR.plot(q_deriv[j],dq_dt[j],'b.', alpha=qual[j])
        discard = dq_dt<=0
        printv('q_deriv = ', q_deriv)
        printv('dq_dt = ', dq_dt)
        printv('sigma=1/Rsq*qual = ', 1./(Rsq*qual))
        printv('discard', discard)
        dq_dt = dq_dt[~discard]
        t_deriv = t_deriv[~discard]
        correction_deriv = correction_deriv[~discard]
        q_deriv = q_deriv[~discard]
        Rsq = Rsq[~discard]
        qual = qual[~discard]
        printv('q_deriv = ', q_deriv)
        printv('dq_dt = ', dq_dt)
        printv('sigma=1./Rsq*qual = ', 1./Rsq*qual)
        #if axR is not None:
            #axR.plot(q_deriv,dq_dt,'b-', lw=0.5, alpha=0.5)
        ## (re-)Fit a power law to the individual recession
        #try:
        if False:
            def func(x, a, b):
                return np.log(a) + b * x
            popt, pcov = curve_fit(func, np.log(q_deriv), np.log(dq_dt*correction_deriv),sigma=1./(Rsq*qual), absolute_sigma=False, p0=(afit, bfit))
            b = popt[1]
            a = popt[0]
        if True:
            def func(x, a, b):
                print('a, b', a, b)
                return a * x**b
            popt, pcov = curve_fit(func, q_deriv, dq_dt*correction_deriv,sigma=1./(Rsq*qual), absolute_sigma=False, p0=(afit, bfit))
            b = popt[1]
            a = popt[0]
        #except:
        #    a = np.NaN
        #    b = np.NaN
        #    pass
        printv('final fit:   ', a, b)
        #if axR is not None:
            #axR.plot(q_deriv,afit*q_deriv**bfit,'r-', lw=0.5, alpha=(Rsq*qual).mean(), zorder=100)
        self.t_deriv = t_deriv
        self.correction_deriv = correction_deriv
        self.q_deriv = q_deriv
        self.dq_dt = dq_dt
        #self.dq_dt_func = lambda qq: np.exp(func(np.log(qq), *popt))
        self.dq_dt_func = lambda qq: func(qq, *popt)
        self.rating = Rsq*qual
        self.a= a
        self.b = b
        self.deriv_fit_x = deriv_fit_x
        self.deriv_fit_y = deriv_fit_y
        self.deriv_fit_ypred = deriv_fit_ypred

def printv(*args):
    if _verbose:
        print(*args)
