import math
import random
import time
import numpy as np
import operator as op
import scipy.stats as si
from functools import reduce, wraps
from scipy.stats import norm

def timethis(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {} milliseconds'.format(func.__module__, func.__name__, (end - start)*1e3))
        return r
    return wrapper

df_dict = {'df_S':100, 
           'df_K':100,
           'df_T':0.25,
           'df_r':0.05,
           'df_b':0.05,
           'df_q':0,
           'df_sigma':0.2,
           'df_option':'call',
           'df_steps':10,
           'df_nodes':100,
           'df_vvol':0.5,
           'df_simulations':10000,
           'df_output_flag':'Price',
           'df_american':False,
           'df_step':10,
           'df_state':10,
           'df_skew':0.0004}

class Pricer():
    
    def __init__(self, S=df_dict['df_S'], K=df_dict['df_K'], T=df_dict['df_T'], r=df_dict['df_r'], 
             q=df_dict['df_q'], sigma=df_dict['df_sigma'], option=df_dict['df_option'], 
             steps=df_dict['df_steps'], nodes=df_dict['df_nodes'], vvol=df_dict['df_vvol'], 
             simulations=df_dict['df_simulations'], output_flag=df_dict['df_output_flag'], 
             american=df_dict['df_american'], step=df_dict['df_step'], state=df_dict['df_state'], 
             skew=df_dict['df_skew']):
        
        #self.S = S # Spot price
        #self.K = K # Strike price
        pass
    
    def _ncr(self, n, r):
        
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom  # or / in Python 2

    @timethis
    def bsm(self, S, K, T, r, q, sigma, option='call'):
        
        b = r - q
        carry = np.exp((b - r) * T)
        d1 = (np.log(S / K) + (b + (0.5 * sigma ** 2)) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (b - (0.5 * sigma ** 2)) * T) / (sigma * np.sqrt(T))
          
        # Cumulative normal distribution function
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        minusNd1 = si.norm.cdf(-d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        minusNd2 = si.norm.cdf(-d2, 0.0, 1.0)
               
        if option == "call":
            opt_price = ((S * carry * Nd1) - 
                              (K * np.exp(-r * T) * Nd2))  
        if option == 'put':
            opt_price = ((K * np.exp(-r * T) * minusNd2) - 
                              (S * carry * minusNd1))
               
        return opt_price
    
    
    @timethis
    def bsm_vega(self, S, K, T, r, q, sigma, option='call'):
            
        b = r - q
        carry = np.exp((b - r) * T)
        d1 = (np.log(S / K) + (b + (0.5 * sigma ** 2)) * T) / (sigma * np.sqrt(T))
        nd1 = (1 / np.sqrt(2 * np.pi)) * (np.exp(-d1 ** 2 * 0.5))
        
        opt_vega = S * carry * nd1 * np.sqrt(T)
         
        return opt_vega
    
    
    @timethis
    def eurobin(self, S, K, T, r, q, sigma, steps, option='call'):
        dt = T / steps
        b = r - q
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(b * dt) - d) / (u - d)
        a = int(np.log(K / (S * (d**steps))) / np.log(u / d)) + 1
        
        val = 0
        
        if option == 'call':
            for j in range(a, steps + 1):
                val = val + (self._ncr(steps, j) * (p ** j) * ((1 - p) ** (steps - j)) * 
                             ((S * (u ** j) * (d ** (steps - j))) - K))
        if option == 'put':
            for j in range(0, a):
                val = val + (self._ncr(steps, j) * (p ** j) * ((1 - p) ** (steps - j)) * 
                             (K - ((S * (u ** j)) * (d ** (steps - j)))))
                               
        return np.exp(-r * T) * val                     
                
    
    @timethis
    def crrbin(self, S, K, T, r, q, sigma, steps, option='call', output_flag='price', american=False):
        
        z = 1
        if option == 'put':
            z = -1
        
        dt = T / steps
        b = r - q
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(b * dt) - d) / (u - d)
        df = np.exp(-r * dt)
        optionvalue = {}
        returnvalue = {}
        
        for i in range(steps + 1):
            optionvalue[i] = max(0, z * (S * (u ** i) * (d ** (steps - i)) - K))
            
            
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                if american == True:
                    optionvalue[i] = ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df
                if american == False:
                    optionvalue[i] = max((z * (S * (u ** i) * (d ** (j - i)) - K)),  
                                         ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df)
            
            if j == 2:
                returnvalue[2] = ((optionvalue[2] - optionvalue[1]) / 
                                  (S * (u ** 2) - S) - (optionvalue[1] - optionvalue[0]) / 
                                   (S - S * (d ** 2))) / (0.5 * (S * (u ** 2) - S * (d ** 2)))
                returnvalue[3] = optionvalue[1]
                
            if j == 1:
                returnvalue[1] = (optionvalue[1] - optionvalue[0]) / (S * u - S * d)
            
        returnvalue[3] = (returnvalue[3] - optionvalue[0]) / (2 * dt) / 365
        returnvalue[0] = optionvalue[0]
        
        if output_flag == 'price':
            result = returnvalue[0]
        if output_flag == 'delta':
            result = returnvalue[1]
        if output_flag == 'gamma':
            result = returnvalue[2]
        if output_flag == 'theta':
            result = returnvalue[3]
        if output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]),
                      'Theta = '+str(returnvalue[3]))
                               
        return result
    
    
    @timethis
    def lrbin(self, S, K, T, r, q, sigma, steps, option='call', output_flag='price', american=False):
        # Leisen Reimer Binomial
        z = 1
        if option == 'put':
            z = -1
        
        b = r - q
        d1 = (np.log(S / K) + (b + (0.5 * sigma ** 2)) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (b - (0.5 * sigma ** 2)) * T) / (sigma * np.sqrt(T))
        hd1 = 0.5 + np.sign(d1) * (0.25 - 0.25 * np.exp(-(d1 / (steps + 1 / 3 + 0.1 / (steps + 1))) ** 2 * (steps + 1 / 6))) ** (0.5)
        hd2 = 0.5 + np.sign(d2) * (0.25 - 0.25 * np.exp(-(d2 / (steps + 1 / 3 + 0.1 / (steps + 1))) ** 2 * (steps + 1 / 6))) ** (0.5)
        
        dt = T / steps
        p = hd2
        u = np.exp(b * dt) * hd1 / hd2
        d = (np.exp(b * dt) - p * u) / (1 - p)
        df = np.exp(-r * dt)
    
        optionvalue = {}
        returnvalue = {}
        
        for i in range(steps + 1):
            optionvalue[i] = max(0, z * (S * (u ** i) * (d ** (steps - i)) - K))
            
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                if american == True:
                    optionvalue[i] = ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df
                if american == False:
                    optionvalue[i] = max((z * (S * (u ** i) * (d ** (j - i)) - K)),  
                                         ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df)
                    
            if j == 2:
                returnvalue[2] = ((optionvalue[2] - optionvalue[1]) / 
                                  (S * (u ** 2) - S * u * d) - (optionvalue[1] - optionvalue[0]) / 
                                   (S * u * d - S * (d ** 2))) / (0.5 * (S * (u ** 2) - S * (d ** 2)))
                returnvalue[3] = optionvalue[1]
                
            if j == 1:
                returnvalue[1] = (optionvalue[1] - optionvalue[0]) / (S * u - S * d)
            
        returnvalue[0] = optionvalue[0]
        
        if output_flag == 'price':
            result = returnvalue[0]
        if output_flag == 'delta':
            result = returnvalue[1]
        if output_flag == 'gamma':
            result = returnvalue[2]
        if output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]))
    
        return result        
    
    
    @timethis
    def tt(self, S, K, T, r, q, sigma, steps, option='call', output_flag='price', american=False):
        # Trinomial Tree
        
        z = 1
        if option == 'put':
            z = -1
        
        dt = T / steps
        b = r - q
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = np.exp(-sigma * np.sqrt(2 * dt))
        pu = ((np.exp(b * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) / 
              (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
        pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(b * dt / 2)) / 
              (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
        pm = 1 - pu - pd
        df = np.exp(-r * dt)
        optionvalue = {}
        returnvalue = {}
        
        for i in range(2 * steps + 1):
            optionvalue[i] = max(0, z * (S * (u ** max(i - steps, 0)) * (d ** (max((steps - i), 0))) - K))
            
            
        for j in range(steps - 1, -1, -1):
            for i in range(j * 2 + 1):
                
                optionvalue[i] = (pu * optionvalue[i + 2] + pm * optionvalue[i + 1] + pd * optionvalue[i]) * df
                
                if american == True:
                    optionvalue[i] = max(z * (S * (u ** max(i - j, 0)) * (d ** (max((j - i), 0))) - K), optionvalue[i])
            
            if j == 1:
                returnvalue[1] = (optionvalue[2] - optionvalue[0]) / (S * u - S * d)
                returnvalue[2] = ((optionvalue[2] - optionvalue[1]) / 
                                  (S * u - S) - (optionvalue[1] - optionvalue[0]) / 
                                   (S - S * d )) / (0.5 * ((S * u) - (S * d)))                              
                returnvalue[3] = optionvalue[0]
                
        #returnvalue[3] = (returnvalue[3] - optionvalue[0]) / dt / 365
        returnvalue[0] = optionvalue[0]
        
        if output_flag == 'price':
            result = returnvalue[0]
        if output_flag == 'delta':
            result = returnvalue[1]
        if output_flag == 'gamma':
            result = returnvalue[2]
        if output_flag == 'theta':
            result = returnvalue[3]
        if output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]),
                      'Theta = '+str(returnvalue[3]))
                               
        return result                     
    
    
    @timethis
    def imptt(self, S, K, T, r, q, sigma, steps, option='call', output_flag='price', 
                             american=False, step_n=3, state_i=2, skew=0.0004):
        # Implied Trinomial Tree
        """
        Return Flag:
            UPM: A matrix of implied up transition probabilities
            DPM: A matrix of implied down transition probabilities
            LVM: A matrix of implied local volatilities
            ADM: A matrix of Arrow-Debreu prices at a single node
            DPni: The implied down transition probability at a single node
            ADni: The Arrow-Debreu price at a single node (at time step step_n and state state_n)
            LVni: The local volatility at a single node
            call: The value of a European call option
            put: The value of a European call option
        """    
        
        z = 1
        if option == 'put':
            z = -1
        
        optionvaluenode = np.zeros((steps * 2 + 1))
        arrowdebreu = np.zeros((steps + 1, steps * 2 + 1), dtype='float')
        upprob = np.zeros((steps, steps * 2 - 1), dtype='float')
        downprob = np.zeros((steps, steps * 2 - 1), dtype='float')
        localvol = np.zeros((steps, steps * 2 - 1), dtype='float')
        
        dt = T / steps
        b = r - q
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1 / u
        df = np.exp(-r * dt)
        arrowdebreu[0, 0] = 1 
                
        for n in range(steps):
            for i in range(n * 2 + 1):
                val = 0
                Si1 = S * (u ** (max(i - n, 0))) * (d ** (max(n * 2 - n - i, 0)))
                Si = Si1 * d
                Si2 = Si1 * u
                Fi = Si1 * np.exp(b * dt)
                sigmai = sigma + (S - Si1) * skew
                if i < (n * 2) / 2 + 1:
                    for j in range(i):
                        Fj = S * (u ** (max(j - n, 0))) * (d ** (max(n * 2 - n - j, 0))) * np.exp(b * dt)
                        val = val + arrowdebreu[n, j] * (Si1 - Fj)
                        
                    optionvalue = self.tt(S=S, K=Si1, T=(n + 1)*dt, r=r, q=0, sigma=sigmai, n=n + 1, option='put', 
                                                output_flag='price', american=False)
        
                    qi = (np.exp(r * dt) * optionvalue - val) / (arrowdebreu[n, i] * (Si1 - Si))
                    pi = (Fi + qi * (Si1 - Si) - Si1) / (Si2 - Si1)
                else:
                    optionvalue = self.tt(S=S, K=Si1, T=(n + 1) * dt, r=r, q=0, sigma=sigmai, n=n + 1, option='call', 
                                                output_flag='price', american=False)
                    val = 0
                    for j in range(i + 1, n * 2 + 1):
                        Fj = S * (u ** (max(j - n, 0))) * (d ** (max(n * 2 - n - j, 0))) * np.exp(b * dt)
                        val = val + arrowdebreu[n, j] * (Fj- Si1)
    
                    pi = (np.exp(r * dt) * optionvalue - val) / (arrowdebreu[n, i] * (Si2 - Si1))
                    qi = (Fi - pi * (Si2 - Si1) - Si1) / (Si - Si1)
                
                # Replacing negative probabilities    
                if pi < 0 or pi > 1 or qi < 0 or qi > 1:
                    if Fi > Si1 and Fi < Si2:
                        pi = 1 / 2 * ((Fi - Si1) / (Si2 - Si1) + (Fi - Si) / (Si2 - Si))
                        qi = 1 / 2 * ((Si2 - Fi) / (Si2 - Si))
                    elif Fi > Si and Fi < Si1:
                        pi = 1 / 2 * ((Fi - Si) / (Si2 - Si))
                        qi = 1 / 2 * ((Si2 - Fi) / (Si2 - Si1) + (Si1 - Fi) / (Si1 - Si))
    
                downprob[n, i] = qi
                upprob[n, i] = pi
                # Calculating local volatilities
                Fo = (pi * Si2 + qi * Si + (1 - pi -qi) * Si1)
                localvol[n, i] = np.sqrt((pi * (Si2 - Fo) ** 2 + (1 - pi - qi) * (Si1 - Fo) ** 2 + qi * (Si - Fo) ** 2) / (Fo ** 2 * dt))
        
                # Calculating Arrow-Debreu prices
                if n == 0:
                    arrowdebreu[n + 1, i] = qi * arrowdebreu[n, i] * df
                    arrowdebreu[n + 1, i + 1] = (1 - pi - qi) * arrowdebreu[n, i] * df
                    arrowdebreu[n + 1, i + 2] = pi * arrowdebreu[n, i] * df
                elif n > 0 and i == 0:
                    arrowdebreu[n + 1, i] = qi * arrowdebreu[n, i] * df
                elif n > 0 and i == n * 2:
                    arrowdebreu[n + 1, i] = (upprob[n, i - 2] * arrowdebreu[n, i - 2] * df + 
                                             (1 - upprob[n, i - 1] - downprob[n, i - 1]) * 
                                             arrowdebreu[n, i - 1] * df + qi * arrowdebreu[n, i] * df)
                    arrowdebreu[n + 1, i + 1] = (upprob[n, i - 1] * arrowdebreu[n, i - 1] * 
                                                 df + (1 - pi - qi) * arrowdebreu[n, i] * df)
                    arrowdebreu[n + 1, i + 2] = pi * arrowdebreu[n, i] * df
                elif n > 0 and i == 1:
                    arrowdebreu[n + 1, i] = ((1 - upprob[n, i - 1] - downprob[n, i - 1]) * 
                                             arrowdebreu[n, i - 1] * df + qi * arrowdebreu[n, i] * df)
                else:
                    arrowdebreu[n + 1, i] = (upprob[n, i - 2] * arrowdebreu[n, i - 2] * df + 
                                             (1 - upprob[n, i - 1] - downprob[n, i - 1]) * 
                                             arrowdebreu[n, i - 1] * df + qi * arrowdebreu[n, i] * df)
    
    
        if output_flag == 'DPM':
            result = downprob
        elif output_flag == 'UPM':    
            result = upprob
        elif output_flag == 'DPni':    
            result = downprob[step_n, state_i]
        elif output_flag == 'UPni':    
            result = upprob[step_n, state_i]        
        elif output_flag == 'ADM':    
            result = arrowdebreu
        elif output_flag == 'LVM': 
            result = localvol
        elif output_flag == 'LVni':
            result = localvol[step_n, state_i]
        elif output_flag == 'ADni':    
            result = arrowdebreu[step_n, state_i]
        elif output_flag == 'Price':
            # Calculation of option price using the implied trinomial tree
            
            for i in range(2 * steps + 1):
                optionvaluenode[i] = max(0, z * (S * (u ** max(i - steps, 0)) * (d ** (max((steps - i), 0))) - K))    
    
            for n in range(steps - 1, -1, -1):
                for i in range(n * 2 + 1):
                    optionvaluenode[i] = ((upprob[n, i] * optionvaluenode[i + 2] + 
                                          (1 - upprob[n, i] - downprob[n, i]) * optionvaluenode[i + 1] + 
                                          downprob[n, i] * optionvaluenode[i]) * df)
    
            result = optionvaluenode[0]         
                               
        return result    
    
    
    @timethis
    def expfd(self, S, K, T, r, q, sigma, nodes, option='call', american=False):
        # Explicit Finite Differences
        z = 1
        if option == 'put':
            z = -1
        
        b = r - q    
        dS = S / nodes
        nodes = int(K / dS) * 2
        St = np.zeros((nodes + 2), dtype='float')
        
        SGridtPt = int(S / dS)
        dt = (dS ** 2) / ((sigma ** 2) * 4 * (K ** 2))
        N = int(T / dt) + 1
        
        C = np.zeros((N + 1, nodes + 2), dtype='float')
        dt = T / N
        Df = 1 / (1 + r * dt)
          
        for i in range(nodes + 1):
            St[i] = i * dS # Asset price at maturity
            C[N, i] = max(0, z * (St[i] - K) ) # At maturity
            
        for j in range(N - 1, -1, -1):
            for i in range(1, nodes):
                pu = 0.5 * ((sigma ** 2) * (i ** 2) + b * i) * dt
                pm = 1 - (sigma ** 2) * (i ** 2) * dt
                pd = 0.5 * ((sigma ** 2) * (i ** 2) - b * i) * dt
                C[j, i] = Df * (pu * C[j + 1, i + 1] + pm * C[j + 1, i] + pd * C[j + 1, i - 1])
                if american == True:
                    C[j, i] = max(z * (St[i] - K), C[j, i])
                    
                if z == 1: # Call option
                    C[j, 0] = 0
                    C[j, nodes] = (St[i] - K)
                else:
                    C[j, 0] = K
                    C[j, nodes] = 0
        
        result = C[0, SGridtPt]
    
        return result          
    
    
    @timethis
    def impfd(self, S, K, T, r, q, sigma, steps, nodes, option='call', american=False):
        # Implicit Finite Differences
        z = 1
        if option == 'put':
            z = -1
        
        b = r - q    
        
        # Make sure current asset price falls at grid point
        dS = 2 * S / nodes
        SGridtPt = int(S / dS)
        nodes = int(K / dS) * 2
        dt = T / steps
        
        CT = np.zeros(nodes + 1)
        p = np.zeros((nodes + 1, nodes + 1), dtype='float')
        
        for j in range(nodes + 1):
            CT[j] = max(0, z * (j * dS - K)) # At maturity
            for i in range(nodes + 1):
                p[j, i] = 0
                
        p[0, 0] = 1
        for i in range(1, nodes):
            p[i, i - 1] = 0.5 * i * (b - (sigma ** 2) * i) * dt
            p[i, i] = 1 + (r + (sigma ** 2) * (i ** 2)) * dt
            p[i, i + 1] = 0.5 * i * (-b - (sigma ** 2) * i) * dt
            
        p[nodes, nodes] = 1
        
        C = np.matmul(np.linalg.inv(p), CT.T)
        
        for j in range(steps - 1, 0, -1):
            C = np.matmul(np.linalg.inv(p), C)
            
            if american == True: # American option
                for i in range(1, nodes + 1):
                    C[i] = max(float(C[i]), z * ((i - 1) * dS - K))
                
        result = C[SGridtPt + 1]
        
        return result, C, p, CT    
    
    
    @timethis
    def expfdlns(self, S, K, T, r, q, sigma, steps, nodes, option='call', american=False):
        # Explicit Finite Differences - rewrite BS-PDE in terms of ln(S) 
        z = 1
        if option == 'put':
            z = -1
        
        b = r - q    
        
        dt = T / steps
        dx = sigma * np.sqrt(3 * dt)
        pu = 0.5 * dt * (((sigma / dx) ** 2) + (b - (sigma ** 2) / 2) / dx)
        pm = 1 - dt * ((sigma / dx) ** 2) - r * dt
        pd = 0.5 * dt * (((sigma / dx) ** 2) - (b - (sigma ** 2) / 2) / dx)
        St = {}
        St[0] = S * np.exp(-nodes / 2 * dx)
        C = np.zeros((int(nodes / 2) + 1, nodes + 2), dtype='float')
        C[steps, 0] = max(0, z * (St[0] - K))
        
        for i in range(1, nodes + 1):
            St[i] = St[i - 1] * np.exp(dx) # Asset price at maturity
            C[steps, i] = max(0, z * (St[i] - K) ) # At maturity
        
        for j in range(steps - 1, -1, -1):
            for i in range(1, nodes):
                C[j, i] = pu * C[j + 1, i + 1] + pm * C[j + 1, i] + pd * C[j + 1, i - 1]
                if american == True:
                    C[j, i] = max(C[j, i], z * (St[i] - K))
                    
                C[j, nodes] = C[j, nodes - 1] + St[nodes] - St[nodes - 1] # Upper boundary
                C[j, 0] = C[j, 1] # Lower boundary
           
        result = C[0, int(nodes / 2)]
    
        return result   
    
    
    @timethis
    def cn(self, S, K, T, r, q, sigma, steps, nodes, option='call', american=False):
        # Crank Nicolson
        
        z = 1
        if option == 'put':
            z = -1
        
        b = r - q    
        
        dt = T / steps
        dx = sigma * np.sqrt(3 * dt)
        pu = -0.25 * dt * (((sigma / dx) ** 2) + (b - (sigma ** 2) / 2) / dx)
        pm = 1 + 0.5 * dt * ((sigma / dx) ** 2) + 0.5 * r * dt
        pd = -0.25 * dt * (((sigma / dx) ** 2) - (b - (sigma ** 2) / 2) / dx)
        St = {}
        pmd = {}
        p = {}
        St[0] = S * np.exp(-nodes / 2 * dx)
        C = np.zeros((int(nodes / 2) + 2, nodes + 2), dtype='float')
        C[0, 0] = max(0, z * (St[0] - K))
        
        for i in range(1, nodes + 1):
            St[i] = St[i - 1] * np.exp(dx) # Asset price at maturity
            C[0, i] = max(0, z * (St[i] - K) ) # At maturity
        
        pmd[1] = pm + pd
        p[1] = -pu * C[0, 2] - (pm - 2) * C[0, 1] - pd * C[0, 0] - pd * (St[1] - St[0])
        
        for j in range(steps - 1, -1, -1):
            for i in range(2, nodes):
                p[i] = -pu * C[0, i + 1] - (pm - 2) * C[0, i] - pd * C[0, i - 1] - p[i - 1] * pd / pmd[i - 1]
                pmd[i] = pm - pu * pd / pmd[i - 1]
    
            for i in range(nodes - 2, 0, -1):
                C[1, i] = (p[i] - pu * C[1, i + 1]) / pmd[i]
                
                for i in range(nodes + 1):
                    C[0, i] = C[1, i]
                    if american == True:
                        C[0, i] = max(C[1, i], z * (St[i] - K))
           
        result = C[0, int(nodes / 2)]
    
        return result   
    
    
    @timethis
    def mc(self, S, K, T, r, q, sigma, simulations, option='call'):
        # Standard Monte Carlo
        b = r - q
        Drift = (b - (sigma ** 2) / 2) * T
        sigmarT = sigma * np.sqrt(T)
        val = 0
        
        z = 1
        if option == 'put':
            z = -1
        
        for i in range(1, simulations + 1):
            St = S * np.exp(Drift + sigmarT * norm.ppf(random.random(), loc=0, scale=1))
            val = val + max(z * (St - K), 0) 
            
        result = np.exp(-r * T) * val / simulations
        
        return result
    
    
    @timethis
    def mcgreeks(self, S, K, T, r, q, sigma, simulations, option='call', output_flag='price'):
        # Standard Monte Carlo with Greeks
        
        b = r - q
        Drift = (b - (sigma ** 2) / 2) * T
        sigmarT = sigma * np.sqrt(T)
        val = 0
        deltasum = 0
        gammasum = 0
        output = {}
        
        z = 1
        if option == 'put':
            z = -1
        
        for i in range(1, simulations + 1):
            St = S * np.exp(Drift + sigmarT * norm.ppf(random.random(), loc=0, scale=1))
            val = val + max(z * (St - K), 0) 
            if z == 1 and St > K:
                deltasum = deltasum + St
            if z == -1 and St < K:
                deltasum = deltasum + St
            if abs(St - K) < 2:
                gammasum = gammasum + 1
                
        # Option Value
        output[0] = np.exp(-r * T) * val / simulations       
            
        # Delta
        output[1] = np.exp(-r * T) * deltasum / (simulations * S)
        
        # Gamma
        output[2] = np.exp(-r * T) * ((K / S) ** 2) * gammasum / (4 * simulations)
        
        # Theta
        output[3] = (r * output[0] - b * S * output[1] - 0.5 * (sigma ** 2) * (S ** 2) * output[2]) / 365
        
        # Vega
        output[4] = output[2] * sigma * (S ** 2) * T / 100
    
        if output_flag == 'price':
            result = output[0]
        if output_flag == 'delta':
            result = output[1]
        if output_flag == 'gamma':
            result = output[2]
        if output_flag == 'theta':
            result = output[3]
        if output_flag == 'vega':
            result = output[4]
        if output_flag == 'all':
            result = ('Price = '+str(output[0]),
                      'Delta = '+str(output[1]),
                      'Gamma = '+str(output[2]),
                      'Theta = '+str(output[3]),
                      'Vega = '+str(output[4]))
                
        return result
    
    
    @timethis
    def hw87sv(self, S, K, T, r, q, sigma, vvol, option='call'):
        # Hull White 1987 Stochastic Volatility
        b = r - q
        
        k = vvol ** 2 * T
        ek = np.exp(k)
        
        d1 = (np.log(S / K) + (b + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
           
        cgbs = self.bsm(S, K, T, r, q, sigma, option='call')
        
        # Partial Derivatives
        cVV = S * np.exp((b - r) * T) * np.sqrt(T) * Nd1 * (d1 * d2 - 1) / (4 * (sigma ** 3))
        cVVV = (S * np.exp((b - r) * T) * np.sqrt(T) * Nd1 * ((d1 * d2 - 1) * (d1 * d2 - 3) - 
                                                              ((d1 ** 2) + (d2 ** 2))) / (8 * (sigma ** 5)))                                                             
        
        callvalue = (cgbs + 1 / 2 * cVV * (2 * sigma ** 4 * (ek - k - 1) / k ** 2 - sigma  ** 4) + 
                     1 / 6 * cVVV * sigma ** 6 * (ek ** 3 - (9 + 18 * k) * ek + 8 + 24 * k + 
                                                  18 * k ** 2 + 6 * k ** 3) / (3 * k ** 3))
        
        if option == 'call':
            result = callvalue
        if option == 'put': # use put-call parity to find put
            result = callvalue - S * np.exp((b - r) * T) + K * np.exp(-r * T)
            
        return result


    def cd(self, R):
        # Cholesky Decomposition
        
        # Number of columns in input correlation matrix R
        n = len(R[0])
        
        a = np.zeros((n + 1, n + 1))
        M = np.zeros((n + 1, n + 1))
        
        for i in range(n + 1):
            for j in range(n + 1):
                a[i, j] = R[i, j]
                M[i, j] = 0
                
        for i in range(n + 1):
            for j in range(n + 1):
                U = a[i, j]
            for h in range(1, i):
                U = U - M[i, h] * M[j, h]
            if j == 1:
                M[i, i] = np.sqrt(U)
            else:
                M[j, i] = U / M[i, i]
        
        return M        



class SABRVolatility():
    
    def __init__(self, F, X, T, ATMvol, Beta, VolVol, rho):
        self.F = F
        self.X = X
        self.T = T
        self.ATMvol = ATMvol
        self.Beta = Beta
        self.VolVol = VolVol
        self.rho = rho
        
        
    def calibrate(self):
        return self._alphasabr(self._findalpha())
    
    
    def _alphasabr(self, Alpha):
        # The SABR skew vol function
        
        dSABR = np.zeros(4)
        dSABR[1] = (Alpha / ((self.F * self.X) ** ((1 - self.Beta) / 2) * (1 + (((1 - self.Beta) ** 2) / 24) * 
                    (np.log(self.F / self.X) ** 2) + ((1 - self.Beta) ** 4 / 1920) * (np.log(self.F / self.X) ** 4))))
        
        if abs(self.F - self.X) > 10 ** -8:
            sabrz = (self.VolVol / Alpha) * (self.F * self.X) ** ((1 - self.Beta) / 2) * np.log(self.F / self.X)
            y = (np.sqrt(1 - 2 * self.rho * sabrz + sabrz ** 2) + sabrz - self.rho) / (1 - self.rho)
            if abs(y - 1) < 10 ** -8:
                dSABR[2] = 1
            elif y > 0:
                dSABR[2] = sabrz / np.log(y)
            else:
                dSABR[2] = 1
        else:
            dSABR[2] = 1
            
        dSABR[3] = (1 + ((((1 - self.Beta) ** 2 / 24) * Alpha ** 2 / ((self.F * self.X) ** (1 - self.Beta))) + 
                         0.25 * self.rho * self.Beta * self.VolVol * Alpha / ((self.F * self.X) ** ((1 - self.Beta) / 2)) + 
                         (2 - 3 * self.rho ** 2) * self.VolVol ** 2 / 24) * self.T)
        
        result = dSABR[1] * dSABR[2] * dSABR[3]
        
        return result
    
    
    def _findalpha(self):
        # Alpha is a function of atm vol etc
        
        result = self._croot((1 - self.Beta) ** 2 * self.T / (24 * self.F **(2 - 2 * self.Beta)), 
                       0.25 * self.rho * self.VolVol * self.Beta * self.T / self.F ** (1 - self.Beta), 
                       1 + (2 - 3 * self.rho ** 2) / 24 * self.VolVol ** 2 * self.T, 
                       -self.ATMvol * self.F ** (1 - self.Beta))
        
        return result
    
    
    def _croot(self, cubic, quadratic, linear, constant):
        
        a = quadratic / cubic
        b = linear / cubic
        C = constant / cubic
        Q = (a ** 2 - 3 * b) / 9
        r = (2 * a ** 3 - 9 * a * b + 27 * C) / 54
        roots = np.zeros(4)
        
        if r ** 2 - Q ** 3 >= 0:
            capA = -np.sign(r) * (abs(r) + np.sqrt(r ** 2 - Q ** 3)) ** (1 / 3)
            if capA == 0:
                capB = 0
            else:
                capB = Q / capA
            result = capA + capB - a / 3
        else:
            theta = self._arccos(r / Q ** 1.5)
            
            # The three roots
            roots[1] = - 2 * np.sqrt(Q) * math.cos(theta / 3) - a / 3
            roots[2] = - 2 * np.sqrt(Q) * math.cos(theta / 3 + 2.0943951023932) - a / 3
            roots[3] = - 2 * np.sqrt(Q) * math.cos(theta / 3 - 2.0943951023932) - a / 3
            
            # locate that one which is the smallest positive root
            # assumes there is such a root (true for SABR model)
            # there is always a small positive root
            
            if roots[1] > 0:
                result = roots[1]
            elif roots[2] > 0:
                result = roots[2]
            elif roots[3] > 0:
                result = roots[3]
        
            if roots[2] > 0 and roots[2] < result:
                result = roots[2]
            
            if roots[3] > 0 and roots[3] < result:
                result = roots[3]
                
        return result
    
    
    def _arccos(self, y):
        result = np.arctan(-y / np.sqrt(-y * y + 1)) + 2 * np.arctan(1)
        
        return result



class ImpliedVol(Pricer):
    
    def __init__(self):
        super().__init__(self)


    @timethis
    def newtonraphson(self, S, K, T, r, q, cm, epsilon, option):
        # Newton-Raphson method - needs knowledge of partial derivative of option 
        # pricing formula with respect to volatility (vega)
        
        # Manaster and Koehler seed value
        vi = np.sqrt(abs(np.log(S / K) + r * T) * 2 / T)
        ci = self.bsm(S, K, T, r, q, vi, option)    
        vegai = self.bsm_vega(S, K, T, r, q, vi)
        minDiff = abs(cm - ci)
    
        while abs(cm - ci) >= epsilon and abs(cm - ci) <= minDiff:
            vi = vi - (ci - cm) / vegai
            ci = self.bsm(S, K, T, r, q, vi, option)
            vegai = self.bsm_vega(S, K, T, r, q, vi)
            minDiff = abs(cm - ci)
            
        if abs(cm - ci) < epsilon:
            result = vi
        else:
            result = 'NA'
        
        return result
    
    
    @timethis
    def bisection(self, S, K, T, r, q, cm, epsilon, option):
        vLow = 0.005
        vHigh = 4
        #epsilon = 1e-08
        cLow = self.bsm(S, K, T, r, q, vLow, option)
        cHigh = self.bsm(S, K, T, r, q, vHigh, option)
        counter = 0
        
        vi = vLow + (cm - cLow) * (vHigh - vLow) / (cHigh - cLow)
        
        while abs(cm - self.bsm(S, K, T, r, q, vi, option)) > epsilon:
            counter = counter + 1
            if counter == 100:
                result = 'NA'
            
            if self.bsm(S, K, T, r, q, vi, option) < cm:
                vLow = vi
            else:
                vHigh = vi
            
            cLow = self.bsm(S, K, T, r, q, vLow, option)
            cHigh = self.bsm(S, K, T, r, q, vHigh, option)
            vi = vLow + (cm - cLow) * (vHigh - vLow) / (cHigh - cLow)
            
        result = vi    
            
        return result

    @timethis
    def iv_naive(self, S, K, T, r, q, cm, epsilon, option):
    
        vi = 0.2
        ci = self.bsm(S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option)
        price_diff = cm - ci
        if price_diff > 0:
            flag = 1
        else:
            flag = -1
        while abs(price_diff) > epsilon:
            while price_diff * flag > 0:
                ci = self.bsm(S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option)
                price_diff = cm - ci
                vi += (0.01 * flag)
            while price_diff * flag < 0:
                ci = self.bsm(S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option)
                price_diff = cm - ci
                vi -= (0.001 * flag)
            while price_diff > 0:
                ci = self.bsm(S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option)
                price_diff = cm - ci
                vi += (0.0001 * flag)
            while price_diff > 0:
                ci = self.bsm(S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option)
                price_diff = cm - ci
                vi -= (0.00001 * flag)
        
        return vi




