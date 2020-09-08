import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


# Initialise default values
df_S = 100 # Spot price
df_S0 = 100
df_SA = np.linspace(80, 120, 100)
df_K = 110 # Strike price
df_T = 0.5 # Time to maturity
df_r = 0.05 # Risk free interest rate
df_b = 0 # Cost of carry
df_q = 0 # Dividend Yield
df_sigma = 0.2 # Volatility
df_eta = 1
df_phi = 1
df_K1 = 95
df_K2 = 100
df_K3 = 105
df_K4 = 110
df_H = 105
df_R = 0
df_T1 = 0.25
df_T2 = 0.25
df_T3 = 0.25
df_T4 = 0.25
df_option = 'call'
df_option1 = 'call'
df_option2 = 'call'
df_option3 = 'call'
df_option4 = 'call'


df_params_list = ['S', 'S0', 'SA', 'K', 'K1', 'K2', 'K3', 'K4', 'H', 'R', 'T', 'T1', 'T2', 
                  'T3', 'T4', 'r', 'b', 'q', 'sigma', 'eta', 'phi', 'option', 'option1', 'option2', 'option3', 
                  'option4']

df_dict = {'df_S':100, 
           'df_S0':100,
           'df_SA':np.linspace(80, 120, 100),
           'df_K':110,
           'df_K1':95,
           'df_K2':105,
           'df_K3':105,
           'df_K4':105,
           'df_H':105,
           'df_R':0,
           'df_T':0.5,
           'df_T1':0.25,
           'df_T2':0.25,
           'df_T3':0.25,
           'df_T4':0.25,
           'df_r':0.05,
           'df_b':0.05,
           'df_q':0,
           'df_sigma':0.2,
           'df_eta':1,
           'df_phi':1,
           'df_option':'call',
           'df_option1':'call',
           'df_option2':'call',
           'df_option3':'call',
           'df_option4':'call'}


class Option():
    
    def __init__(self, S=df_dict['df_S'], S0=df_dict['df_S0'], SA=df_dict['df_SA'], 
                 K=df_dict['df_K'], K1=df_dict['df_K1'], K2=df_dict['df_K2'], K3=df_dict['df_K3'], 
                 K4=df_dict['df_K4'], H=df_dict['df_H'], R=df_dict['df_R'], T=df_dict['df_T'], 
                 T1=df_dict['df_T1'], T2=df_dict['df_T2'], T3=df_dict['df_T3'], 
                 T4=df_dict['df_T4'],r=df_dict['df_r'], b=df_dict['df_b'], 
                 q=df_dict['df_q'], sigma=df_dict['df_sigma'], eta=df_dict['df_eta'], 
                 phi=df_dict['df_phi'], option=df_dict['df_option'], option1=df_dict['df_option1'], 
                 option2=df_dict['df_option2'], option3=df_dict['df_option3'], 
                 option4=df_dict['df_option4'], df_dict=df_dict):

        self.S = S
        self.S0 = S0
        self.SA = SA
        self.K = K
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.H = H # Barrier level
        self.R = R # Rebate
        self.T = T
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.r = r
        self.q = q
        self.b = self.r - self.q
        self.sigma = sigma
        self.eta = eta
        self.phi = phi
        self.option = option
        self.option1 = option1
        self.option2 = option2
        self.option3 = option3
        self.option4 = option4
        self.df_dict = df_dict
                
    
    
    def _initialise_func(self, **kwargs):
        
        #self._reset_params()
        self._refresh_params(**kwargs)
        #self._refresh_dist()
        
        return self

    def _initialise_barriers(self, **kwargs):
        self._refresh_params(**kwargs)
        self._refresh_dist()
        self._barrier_factors()

        return self


    def _reset_params(self, df_dict=df_dict):
        #for key, value in df_dict.items():
        #    self.key = value
        self.S = df_dict['df_S']
        self.S0 = df_dict['df_S0']
        self.SA = df_dict['df_SA']
        self.K = df_dict['df_K']
        self.K1 = df_dict['df_K1']
        self.K2 = df_dict['df_K2']
        self.K3 = df_dict['df_K3']
        self.K4 = df_dict['df_K4']
        self.H = df_dict['df_H']
        self.R = df_dict['df_R']
        self.T = df_dict['df_T']
        self.T1 = df_dict['df_T1']
        self.T2 = df_dict['df_T2']
        self.T3 = df_dict['df_T3']
        self.T4 = df_dict['df_T4']
        self.r = df_dict['df_r']
        self.q = df_dict['df_q']
        self.b = self.r - self.q
        self.sigma = df_dict['df_sigma']
        self.eta = df_dict['df_eta']
        self.phi = df_dict['df_phi']
        self.option = df_dict['df_option']
        self.option1 = df_dict['df_option1']
        self.option2 = df_dict['df_option2']
        self.option3 = df_dict['df_option3']
        self.option4 = df_dict['df_option4']
                
        return self    
    

    def _refresh_params_2(self, **kwargs):
        for k, v in kwargs.items():
            if k is None:
                k = self.df_dict['df_'+k]
            else:
                self.k = k
 
    
    def _refresh_params_3(self, S=None, S0=None, SA=None, K=None, K1=None, K2=None, 
                        K3=None, K4=None, H=None, R=None, T=None, T1=None, T2=None, 
                        T3=None, T4=None, r=None, b=None, q=None, sigma=None, eta=None, 
                        phi=None, option=None, option1=None, option2=None, option3=None, 
                        option4=None):
        arg_dict = locals()
        for k, v in arg_dict.items():
            if k is None:
                k = self.df_dict['df_'+k]
            else:
                self.k = k
    
        self.b = self.r - self.q
        
        self.carry = np.exp((self.b - self.r) * self.T)
        
                
        return self
    
    
    def _refresh_params(self, S=None, S0=None, SA=None, K=None, K1=None, K2=None, 
                        K3=None, K4=None, H=None, R=None, T=None, T1=None, T2=None, 
                        T3=None, T4=None, r=None, b=None, q=None, sigma=None, eta=None, 
                        phi=None, option=None, option1=None, option2=None, option3=None, 
                        option4=None):        
        #arg_dict = locals()
        #arg_list = [] 
        #for k, v in arg_dict.items():
        #    arg_list.append(k)
            
        #global val    
        #for val in arg_list:
        #    if val is None:
        #        val = self.val
        #    else:
        #        self.val = val
        
        if S is None:
            S = df_dict['df_S']
        else:
            self.S = S
        if S0 is None:
            S0 = df_dict['df_S0']
        else:
            self.S0 = S0
        if SA is None:
            SA = df_dict['df_SA']
        else:
            self.SA = SA
        if K is None:
            K = df_dict['df_K']
        else:
            self.K = K
        if K1 is None:
            K1 = df_dict['df_K1']
        else:
            self.K1 = K1
        if K2 is None:
            K2 = df_dict['df_K2']
        else:
            self.K2 = K2
        if K3 is None:
            K3 = df_dict['df_K3']
        else:
            self.K3 = K3
        if K4 is None:
            K4 = df_dict['df_K4']
        else:
            self.K4 = K4
        if H is None:
            H = df_dict['df_H']
        else:
            self.H = H
        if R is None:
            R = df_dict['df_R']
        else:
            self.R = R
        if T is None:
            T = df_dict['df_T']
        else:
            self.T = T
        if T1 is None:
            T1 = df_dict['df_T1']
        else:
            self.T1 = T1
        if T2 is None:
            T2 = df_dict['df_T2']
        else:
            self.T2 = T2
        if T3 is None:
            T3 = df_dict['df_T3']
        else:
            self.T3 = T3    
        if T4 is None:
            T4 = df_dict['df_T4']
        else:
            self.T4 = T4    
        if r is None:
            r = df_dict['df_r']
        else:
            self.r = r
        if q is None:
            q = df_dict['df_q']
        else:
            self.q = q    
        if sigma is None:
            sigma = df_dict['df_sigma']
        else:
            self.sigma = sigma
        if eta is None:
            eta = df_dict['df_eta']
        else:
            self.eta = eta
        if phi is None:
            phi = df_dict['df_phi']
        else:
            self.phi = phi    
        if option is None:
            option = df_dict['df_option']
        else:
            self.option = option
        if option1 is None:
            option1 = df_dict['df_option1']
        else:
            self.option1 = option1
        if option2 is None:
            option2 = df_dict['df_option2']
        else:
            self.option2 = option2
        if option3 is None:
            option3 = df_dict['df_option3']
        else:
            self.option3 = option3
        if option4 is None:
            option4 = df_dict['df_option4']
        else:
            self.option4 = option4
        
        self.b = self.r - self.q
        
        self.carry = np.exp((self.b - self.r) * self.T)
        self.discount = np.exp(-self.r * self.T)
        
        with np.errstate(divide='ignore'):
            self.d1 = (np.log(self.S / self.K) + (self.b + (0.5 * self.sigma ** 2)) * 
                       self.T) / (self.sigma * np.sqrt(self.T))
            
            self.d2 = (np.log(self.S / self.K) + (self.b - (0.5 * self.sigma ** 2)) * 
                       self.T) / (self.sigma * np.sqrt(self.T))
            
            # standardised normal density function
            self.nd1 = (1 / np.sqrt(2 * np.pi)) * (np.exp(-self.d1 ** 2 * 0.5))
            
            # Cumulative normal distribution function
            self.Nd1 = si.norm.cdf(self.d1, 0.0, 1.0)
            self.minusNd1 = si.norm.cdf(-self.d1, 0.0, 1.0)
            self.Nd2 = si.norm.cdf(self.d2, 0.0, 1.0)
            self.minusNd2 = si.norm.cdf(-self.d2, 0.0, 1.0)
                    
        self.params_dict = {'S':self.S, 
                            'S0':self.S0,
                            'SA':self.SA,
                            'K':self.K,
                            'K1':self.K1,
                            'K2':self.K2,
                            'K3':self.K3,
                            'K4':self.K4,
                            'H':self.H,
                            'R':self.R,
                            'T':self.T,
                            'T1':self.T1,
                            'T2':self.T2,
                            'T3':self.T3,
                            'T4':self.T4,
                            'r':self.r,
                            'b':self.b,
                            'q':self.q,
                            'sigma':self.sigma,
                            'eta':self.eta,
                            'phi':self.phi,
                            'option':self.option,
                            'option1':self.option1,
                            'option2':self.option2,
                            'option3':self.option3,
                            'option4':self.option4,
                            'carry': self.carry,
                            'discount':self.discount,
                            'd1':self.d1,
                            'd2':self.d2,
                            'nd1':self.nd1,
                            'Nd1':self.Nd1,
                            'minusNd1':self.minusNd1,
                            'Nd2':self.Nd2,
                            'minusNd2':self.minusNd2}
                        
        return self    

    
    def _barrier_factors(self):
        
        self.mu = (self.b - ((self.sigma ** 2) / 2)) / (self.sigma ** 2)
        self.lamb_da = (np.sqrt(self.mu ** 2 + ((2 * self.r) / self.sigma ** 2)))
        self.z = ((np.log(self.H / self.S) / (self.sigma * np.sqrt(self.T))) + 
                  (self.lamb_da * self.sigma * np.sqrt(self.T)))
        
        self.x1 = (np.log(self.S / self.K) / (self.sigma * np.sqrt(self.T)) + 
                   ((1 + self.mu) * self.sigma * np.sqrt(self.T)))
        
        self.x2 = (np.log(self.S / self.H) / (self.sigma * np.sqrt(self.T)) + 
                   ((1 + self.mu) * self.sigma * np.sqrt(self.T)))
        
        self.y1 = (np.log((self.H ** 2) / (self.S * self.K)) / (self.sigma * np.sqrt(self.T)) + 
                   ((1 + self.mu) * self.sigma * np.sqrt(self.T)))
        
        self.y2 = (np.log(self.H / self.S) / (self.sigma * np.sqrt(self.T)) + 
                   ((1 + self.mu) * self.sigma * np.sqrt(self.T)))
        
        self.carry = np.exp((self.b - self.r) * self.T)
        
        self.A = ((self.phi * self.S * self.carry * si.norm.cdf((self.phi * self.x1), 0.0, 1.0)) - 
                  (self.phi * self.K * np.exp(-self.r * self.T) * 
                   si.norm.cdf(((self.phi * self.x1) - (self.phi * self.sigma * np.sqrt(self.T))), 0.0, 1.0)))

        self.B = ((self.phi * self.S * self.carry * si.norm.cdf((self.phi * self.x2), 0.0, 1.0)) - 
                  (self.phi * self.K * np.exp(-self.r * self.T) * 
                   si.norm.cdf(((self.phi * self.x2) - (self.phi * self.sigma * np.sqrt(self.T))), 0.0, 1.0)))
        
        self.C = ((self.phi * self.S * self.carry * ((self.H / self.S) ** (2 * (self.mu + 1))) * 
                   si.norm.cdf((self.eta * self.y1), 0.0, 1.0)) -  
                  (self.phi * self.K * np.exp(-self.r * self.T) * ((self.H / self.S) ** (2 * self.mu)) * 
                   si.norm.cdf(((self.eta * self.y1) - (self.eta * self.sigma * np.sqrt(self.T))), 0.0, 1.0)))
        
        self.D = ((self.phi * self.S * self.carry * ((self.H / self.S) ** (2 * (self.mu + 1))) * 
                   si.norm.cdf((self.eta * self.y2), 0.0, 1.0)) -  
                  (self.phi * self.K * np.exp(-self.r * self.T) * ((self.H / self.S) ** (2 * self.mu)) * 
                   si.norm.cdf(((self.eta * self.y2) - (self.eta * self.sigma * np.sqrt(self.T))), 0.0, 1.0)))
    
        self.E = ((self.R * np.exp(-self.r * self.T)) * 
                  (si.norm.cdf(((self.eta * self.x2) - (self.eta * self.sigma * np.sqrt(self.T))), 0.0, 1.0) - 
                   (((self.H / self.S) ** (2 * self.mu)) * 
                    si.norm.cdf(((self.eta * self.y2) - (self.eta * self.sigma * np.sqrt(self.T))), 0.0, 1.0))))
        
        self.F = (self.R *
                  (((self.H / self.S) ** (self.mu + self.lamb_da)) *
                  (si.norm.cdf((self.eta * self.z), 0.0, 1.0)) + 
                   (((self.H / self.S) ** (self.mu - self.lamb_da)) * 
                    si.norm.cdf(((self.eta * self.z) - (2 * self.eta * self.lamb_da * self.sigma * np.sqrt(self.T))), 0.0, 1.0))))

        return self


    def price(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option='call'):
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        
        if option == "call":
            self.opt_price = ((self.S * self.carry * self.Nd1) - 
                              (self.K * np.exp(-self.r * self.T) * self.Nd2))  
        if option == 'put':
            self.opt_price = ((self.K * np.exp(-self.r * self.T) * self.minusNd2) - 
                              (self.S * self.carry * self.minusNd1))
        
        np.nan_to_num(self.opt_price, copy=False)
                
        return self.opt_price


    def delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option='call'):
                    
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
                        
        if option == 'call':
            self.opt_delta = self.carry * self.Nd1
        if option == 'put':
            self.opt_delta = -self.carry * self.minusNd1
            
        return self.opt_delta
    
    
    def shift_delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option='call', 
                    shift=25, shift_type='avg'):
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        
        down_shift = self.S-(shift/10000)*self.S
        up_shift = self.S+(shift/10000)*self.S
        opt_price = self.price(S=self.S, K=self.K, T=self.T, r=self.r, q=self.q, 
                               sigma=self.sigma, option=self.option)
        op_shift_down = self.price(S=down_shift, K=self.K, T=self.T, r=self.r, 
                                   q=self.q, sigma=self.sigma, option=self.option)
        op_shift_up = self.price(S=up_shift, K=self.K, T=self.T, r=self.r, q=self.q, 
                                 sigma=self.sigma, option=self.option)
                
        if shift_type == 'up':
            self.opt_delta_shift = (op_shift_up - opt_price) * 4
        if shift_type == 'down':
            self.opt_delta_shift = (opt_price - op_shift_down) * 4
        if shift_type == 'avg':    
            self.opt_delta_shift = (op_shift_up - op_shift_down) * 2
        
        return self.opt_delta_shift
    
    
    def theta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option='call'):
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
    
        if option == 'call':
            self.opt_theta = ((-self.S * self.carry * self.nd1 * self.sigma ) / 
                              (2 * np.sqrt(self.T)) - (self.b - self.r) * self.S * self.carry * 
                              self.Nd1 - self.r * self.K * np.exp(-self.r * self.T) * self.Nd2)
        if option == 'put':   
            self.opt_theta = ((-self.S * self.carry * self.nd1 * self.sigma ) / 
                              (2 * np.sqrt(self.T)) + (self.b - self.r) * self.S * self.carry * 
                              self.minusNd1 + self.r * self.K * np.exp(-self.r * self.T) * self.minusNd2)

        return self.opt_theta
    
    
    def gamma(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # how much delta will change due to a small change in the underlying asset price        
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_gamma = ((self.nd1 * self.carry) / (self.S * self.sigma * np.sqrt(self.T)))
        
        return self.opt_gamma
    
    
    def vega(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

        self.opt_vega = self.S * self.carry * self.nd1 * np.sqrt(self.T)
        
        return self.opt_vega
    
    
    def rho(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option='call'):
                
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        
        if option == 'call':
            self.opt_rho = self.T * self.K * np.exp(-self.r * self.T) * self.Nd2
        if option == 'put':
            self.opt_rho = -self.T * self.K * np.exp(-self.r * self.T) * self.minusNd2
            
        return self.opt_rho


    def vanna(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # aka DdeltaDvol, DvegaDspot 
        # how much delta will change due to a small change in volatility
        # how much vega will change due to a small change in the asset price        
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_vanna = ((-self.carry * self.d2) / self.sigma) * self.nd1 

        return self.opt_vanna               
           

    def charm(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option='call'):
        # aka DdeltaDtime, Delta Bleed 
        # how much delta will change due to a small change in time        
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        
        if option == 'call':
            self.opt_charm = (-self.carry * ((self.nd1 * ((self.b / (self.sigma * np.sqrt(self.T))) - 
                                                          (self.d2 / (2 * self.T)))) + 
                                             ((self.b - self.r) * self.Nd1)))
        if option == 'put':
            self.opt_charm = (-self.carry * ((self.nd1 * ((self.b / (self.sigma * np.sqrt(self.T))) - 
                                                          (self.d2 / (2 * self.T)))) - 
                                             ((self.b - self.r) * self.minusNd1)))
        return self.opt_charm
               

    def zomma(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # DgammaDvol
        # how much gamma will change due to a small change in volatility
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_zomma = (self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma) * 
                          ((self.d1 * self.d2 - 1) / self.sigma))
        
        return self.opt_zomma


    def speed(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # DgammaDspot
        # how much gamma will change due to a small change in the asset price
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_speed = -(self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma) * 
                           (1 + (self.d1 / (self.sigma * np.sqrt(self.T)))) / self.S)
        
        return self.opt_speed


    def color(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # DgammaDtime, gamma bleed
        # how much gamma will change due to a small change in time
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_color = (self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma) * 
                           ((self.r - self.b) + ((self.b * self.d1) / (self.sigma * np.sqrt(self.T))) + 
                            ((1 - self.d1 * self.d2) / (2 * self.T))))
        
        return self.opt_color


    def vomma(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # DvegaDvol, vega convexity, volga
        # how much vega will change due to a small change in implied volatility
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_vomma = (self.vega(self.S, self.K, self.T, self.r, self.q, self.sigma) * 
                           ((self.d1 * self.d2) / (self.sigma)))
        
        return self.opt_vomma


    def ultima(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # DvommaDvol
        # how much vomma will change due to a small change in implied volatility
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_ultima = (self.vomma(self.S, self.K, self.T, self.r, self.q, self.sigma) * 
                           ((1 / self.sigma) * (self.d1 * self.d2 - (self.d1 / self.d2) - 
                                                (self.d2 / self.d1) - 1)))
        
        return self.opt_ultima


    def vega_bleed(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # DvegaDtime
        # how much vega will change due to a small change in time
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_vega_bleed = (self.vega(self.S, self.K, self.T, self.r, self.q, self.sigma) * 
                           (self.r - self.b + ((self.b * self.d1) / (self.sigma * np.sqrt(self.T))) - 
                            ((1 + (self.d1 * self.d2) ) / (2 * self.T))))

        return self.opt_vega_bleed



    def barrier_price(self, S=None, K=None, H=None, R=None, T=None, r=None, q=None, 
                       sigma=None, direction='down', knock='in', option='call'):
        
        self._initialise_barriers(S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                                  option=option)
        
        if direction == 'down' and knock == 'in' and option == 'call':
            self.eta = 1
            self.phi = 1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.C + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.B + self.D + self.E
            

        if direction == 'up' and knock == 'in' and option == 'call':
            self.eta = -1
            self.phi = 1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.A + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.B - self.C + self.D + self.E


        if direction == 'down' and knock == 'in' and option == 'put':
            self.eta = 1
            self.phi = -1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.B - self.C + self.D + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.A + self.E
                
         
        if direction == 'up' and knock == 'in' and option == 'put':
            self.eta = -1
            self.phi = -1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.B + self.D + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.C + self.E
                

        if direction == 'down' and knock == 'out' and option == 'call':
            self.eta = 1
            self.phi = 1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.C + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.B - self.D + self.F
            

        if direction == 'up' and knock == 'out' and option == 'call':
            self.eta = -1
            self.phi = 1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.B + self.C - self.D + self.F


        if direction == 'down' and knock == 'out' and option == 'put':
            self.eta = 1
            self.phi = -1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.B + self.C - self.D + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.F
                
         
        if direction == 'up' and knock == 'out' and option == 'put':
            self.eta = -1
            self.phi = -1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.B - self.D + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.C + self.F

        return self.opt_barrier_payoff    



    def _vis_payoff(self, S0, SA, payoff, label, title='Option Payoff', payoff2=None, 
                    label2=None, payoff3=None, label3=None, payoff4=None, label4=None, 
                    xlabel='Stock Price', ylabel='P&L'):
       
        fig, ax = plt.subplots()
        ax.plot(SA, payoff, color='blue', label=label)
        if payoff2 is not None:
            ax.plot(SA, payoff2, color='red', label=label2)
        if payoff3 is not None:
            ax.plot(SA, payoff3, color='green', label=label3)
        if payoff4 is not None:
            ax.plot(SA, payoff4, color='purple', label=label4)
        ax.axhline(y=0, linewidth=0.5, color='k')
        ax.axvline(x=S0, linewidth=0.5, color='k')
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('1')          
        plt.style.use('seaborn-darkgrid')
        plt.grid(True)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend()
        plt.show()
    
    
    def _vis_greeks_mpl(self, S0, xarray1=None, xarray2=None, xarray3=None, 
                        xarray4=None, yarray1=None, yarray2=None, yarray3=None, 
                        yarray4=None, label1=None, label2=None, label3=None, label4=None, 
                        xlabel=None, ylabel=None):
        
        fig, ax = plt.subplots()
        ax.plot(xarray1, yarray1, color='blue', label=label1)
        ax.plot(xarray2, yarray2, color='red', label=label2)
        ax.plot(xarray3, yarray3, color='green', label=label3)
        if label4 is not None:
            ax.plot(xarray4, yarray4, color='orange', label=label4)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
   
    
    def greeks_graphs_3D(self, S0=100, r=0.01, q=0, sigma=0.2, greek='delta', option='call', 
                         interactive=False, notebook=True, colorscheme='BlueRed'):

        self.TA_lower = 0.01

        if greek == 'price':
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 1
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.price(S=self.x, K=S0, T=self.y, r=r, sigma=sigma, option=option)

        if greek == 'delta':
            self.SA_lower = 0.25
            self.SA_upper = 1.75
            self.TA_upper = 2
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.delta(S=self.x, K=S0, T=self.y, r=r, sigma=sigma, option=option)

        if greek == 'gamma':
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.gamma(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)

        if greek == 'vega':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vega(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)

        if greek == 'theta':    
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 1
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.theta(S=self.x, K=S0, T=self.y, r=r, sigma=sigma, option=option)
            
        if greek == 'rho':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.rho(S=self.x, K=S0, T=self.y, r=r, sigma=sigma, option=option)    

        if greek == 'vomma':               
            option = 'Call / Put'
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vomma(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)

        if greek == 'vanna':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vanna(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)

        if greek == 'zomma':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.zomma(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)
            
        if greek == 'speed':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.speed(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)    

        if greek == 'color':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.color(S=self.x, K=S0, T=self.y, r=r, sigma=sigma) 
            
        if greek == 'ultima':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.ultima(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)     

        if greek == 'vega bleed':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vega_bleed(S=self.x, K=S0, T=self.y, r=r, sigma=sigma)   

        if greek == 'charm':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.25
            self.SA = np.linspace(self.SA_lower * S0, self.SA_upper * S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.charm(S=self.x, K=S0, T=self.y, r=r, sigma=sigma, option=option)

        if option == 'Call / Put':
            titlename = str(option+' Option '+str(greek.title()))
        else:    
            titlename = str(str(option.title())+' Option '+str(greek.title()))


        if interactive == False:
        
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.x,
                            self.y,
                            self.z,
                            rstride=2, cstride=2,
                            cmap=cm.jet,
                            alpha=0.7,
                            linewidth=0.25)
            ax.set_zlim(auto=True)
            ax.invert_xaxis()
            ax.set_xlabel('Underlying Value', fontsize=12)
            ax.set_ylabel('Time to Expiration', fontsize=12)
            ax.set_zlabel(greek, fontsize=12)
            ax.set_title(titlename, fontsize=14)
            plt.show()


        if interactive == True:
            
            contour_x_start = self.TA_lower
            contour_x_stop = self.TA_upper * 360
            contour_x_size = contour_x_stop / 18
            contour_y_start = self.SA_lower
            contour_y_stop = self.SA_upper
            contour_y_size = int((self.SA_upper - self.SA_lower) / 20)
            contour_z_start = np.min(self.z)
            contour_z_stop = np.max(self.z)
            contour_z_size = int((np.max(self.z) - np.min(self.z)) / 10)
            
            
            fig = go.Figure(data=[go.Surface(x=self.y*365, 
                                             y=self.x, 
                                             z=self.z, 
                                             colorscale=colorscheme, 
                                             contours = {"x": {"show": True, "start": contour_x_start, 
                                                               "end": contour_x_stop, "size": contour_x_size, "color":"white"},            
                                                         "y": {"show": True, "start": contour_y_start, 
                                                               "end": contour_y_stop, "size": contour_y_size, "color":"white"},  
                                                         "z": {"show": True, "start": contour_z_start, 
                                                               "end": contour_z_stop, "size": contour_z_size}},)])
            
            camera = dict(
                eye=dict(x=2, y=1, z=1)
            )
            
            
            
            
            fig.update_scenes(xaxis_autorange="reversed")
            fig.update_scenes(yaxis_autorange="reversed")
            fig.update_layout(scene = dict(
                                xaxis = dict(
                                     backgroundcolor="rgb(200, 200, 230)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",),
                                yaxis = dict(
                                    backgroundcolor="rgb(230, 200,230)",
                                    gridcolor="white",
                                    showbackground=True,
                                    zerolinecolor="white"),
                                zaxis = dict(
                                    backgroundcolor="rgb(230, 230,200)",
                                    gridcolor="white",
                                    showbackground=True,
                                    zerolinecolor="white",),
                                xaxis_title='Time to Expiration (Days)',
                                yaxis_title='Underlying Value',
                                zaxis_title=greek,),
                              title=titlename, autosize=False, 
                              width=800, height=800,
                              margin=dict(l=65, r=50, b=65, t=90),
                             scene_camera=camera)
            
            if notebook == True:
                fig.show()
            else:
                plot(fig, auto_open=True)
 
    
    def greeks_graphs_2D(self, x='delta', y='price', S0=100, T=0.25, T1=0.25, T2=0.5, 
                         r=0.01, q=0, sigma=0.2, option='call'):
                
        if x == 'value':
            if y == 'price':
                self.value_price(S0=S0, T=T, r=r, q=q, sigma=sigma, option=option)
            if y == 'vol':
                self.value_vol(S0=S0, T=T, r=r, q=q, option=option)
            if y == 'time':
                self.value_time(S0=S0, r=r, q=q, sigma=sigma, option=option)
        
        if x == 'delta':
            if y == 'price':
                self.delta_price(S0=S0, T=T, r=r, q=q, sigma=sigma, option=option)
            if y == 'vol':
                self.delta_vol(S0=S0, T=T, r=r, q=q, option=option)
            if y == 'time':
                self.delta_time(S0=S0, r=r, q=q, sigma=sigma, option=option)
        
        if x == 'gamma':
            if y == 'price':
                self.gamma_price(S0=S0, T=T, r=r, q=q, sigma=sigma)
            if y == 'vol':
                self.gamma_vol(S0=S0, T=T, r=r, q=q)
            if y == 'time':
                self.gamma_time(S0=S0, r=r, q=q, sigma=sigma)
        
        if x == 'vega':
            if y == 'price':
                self.vega_price(S0=S0, T=T, r=r, q=q, sigma=sigma)
            if y == 'vol':
                self.vega_vol(S0=S0, T=T, r=r, q=q)
            if y == 'time':
                self.vega_time(S0=S0, r=r, q=q, sigma=sigma)  
        
        if x == 'theta':
            if y == 'price':
                self.theta_price(S0=S0, T=T, r=r, q=q, sigma=sigma, option=option)
            if y == 'vol':
                self.theta_vol(S0=S0, T=T, r=r, q=q, option=option)
            if y == 'time':
                self.theta_time(S0=S0, r=r, q=q, sigma=sigma, option=option)
        
        if x == 'rho':
            if y == 'price':
                self.rho_price(S0=S0, T1=T1, T2=T2, r=r, q=q, sigma=sigma)
            if y == 'vol':
                self.rho_vol(S0=S0, T1=T1, T2=T2, r=r, q=q)
    
    
    def value_price(self, S0=100, T=0.25, r=0.05, q=0, sigma=0.2, option='call'):
        
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)
        
        self.C1 = self.price(S=self.SA, K=S0 * 0.9, T=T, r=r, q=q, sigma=sigma, option=option)
        self.C2 = self.price(S=self.SA, K=S0, T=T, r=r, q=q, sigma=sigma, option=option)
        self.C3 = self.price(S=self.SA, K=S0 * 1.1, T=T, r=r, q=q, sigma=sigma, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Underlying Price'
        self.ylabel = 'Theoretical Value'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)        
    
    
    def value_vol(self, S0=100, T=0.25, r=0.05, q=0, option='call'):
        
        self.sigmaA = np.linspace(0.05, 0.5, 100)
        
        self.C1 = self.price(S=S0, K=S0 * 0.9, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C2 = self.price(S=S0, K=S0, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C3 = self.price(S=S0, K=S0 * 1.1, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Volatility %'
        self.ylabel = 'Theoretical Value'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)     
        
    
    def value_time(self, S0=100, r=0.01, q=0, sigma=0.2, option='call'):
        
        self.TA = np.linspace(0.01, 1, 100)
        
        self.C1 = self.price(S=S0, K=S0 * 0.9, T=self.TA, r=r, q=q, sigma=sigma, option=option)
        self.C2 = self.price(S=S0, K=S0, T=self.TA, r=r, q=q, sigma=sigma, option=option)
        self.C3 = self.price(S=S0, K=S0 * 1.1, T=self.TA, r=r, q=q, sigma=sigma, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Time to Expiration (days)'
        self.ylabel = 'Theoretical Value'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)     
    
    
    
    def delta_price(self, S0=100, T=0.25, r=0.05, q=0, sigma=0.2, option='call'):
        
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)
        
        self.C1 = self.delta(S=self.SA, K=S0 * 0.9, T=T, r=r, q=q, sigma=sigma, option=option)
        self.C2 = self.delta(S=self.SA, K=S0, T=T, r=r, q=q, sigma=sigma, option=option)
        self.C3 = self.delta(S=self.SA, K=S0 * 1.1, T=T, r=r, q=q, sigma=sigma, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Underlying Price'
        self.ylabel = 'Delta'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)       
    
    
    def delta_vol(self, S0=100, T=0.25, r=0.05, q=0, option='call'):
        
        self.sigmaA = np.linspace(0.05, 0.5, 100)
        
        self.C1 = self.delta(S=S0, K=S0 * 0.9, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C2 = self.delta(S=S0, K=S0, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C3 = self.delta(S=S0, K=S0 * 1.1, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Volatility %'
        self.ylabel = 'Delta'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)        
        
        
    def delta_time(self, S0=100, r=0.01, q=0, sigma=0.2, option='call'):
        
        self.TA = np.linspace(0.01, 1, 100)
        
        self.C1 = self.delta(S=S0, K=S0 * 0.9, T=self.TA, r=r, q=q, sigma=sigma, option=option)
        self.C2 = self.delta(S=S0, K=S0, T=self.TA, r=r, q=q, sigma=sigma, option=option)
        self.C3 = self.delta(S=S0, K=S0 * 1.1, T=self.TA, r=r, q=q, sigma=sigma, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Time to Expiration (days)'
        self.ylabel = 'Delta'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)     
    
        
    def gamma_price(self, S0=100, T=0.25, r=0.05, q=0, sigma=0.2):
        
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)
        
        self.C1 = self.gamma(S=self.SA, K=S0 * 0.9, T=T, r=r, q=q, sigma=sigma)
        self.C2 = self.gamma(S=self.SA, K=S0, T=T, r=r, q=q, sigma=sigma)
        self.C3 = self.gamma(S=self.SA, K=S0 * 1.1, T=T, r=r, q=q, sigma=sigma)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Underlying Price'
        self.ylabel = 'Gamma'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)        
        
        
    def gamma_vol(self, S0=100, T=0.25, r=0.05, q=0):

        self.sigmaA = np.linspace(0.05, 0.5, 100)
        
        self.C1 = self.gamma(S=S0, K=S0 * 0.9, T=T, r=r, q=q, sigma=self.sigmaA)
        self.C2 = self.gamma(S=S0, K=S0, T=T, r=r, q=q, sigma=self.sigmaA)
        self.C3 = self.gamma(S=S0, K=S0 * 1.1, T=T, r=r, q=q, sigma=self.sigmaA)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Volatility %'
        self.ylabel = 'Gamma'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)            
        
    
    def gamma_time(self, S0=100, r=0.01, q=0, sigma=0.2):
        
        self.TA = np.linspace(0.01, 1, 100)
        
        self.C1 = self.gamma(S=S0, K=S0 * 0.9, T=self.TA, r=r, q=q, sigma=sigma)
        self.C2 = self.gamma(S=S0, K=S0, T=self.TA, r=r, q=q, sigma=sigma)
        self.C3 = self.gamma(S=S0, K=S0 * 1.1, T=self.TA, r=r, q=q, sigma=sigma)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Time to Expiration (days)'
        self.ylabel = 'Gamma'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)     
    
    
    def vega_price(self, S0=100, T=0.25, r=0.05, q=0, sigma=0.2):
        
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)
        
        self.C1 = self.vega(S=self.SA, K=S0 * 0.9, T=T, r=r, q=q, sigma=sigma)
        self.C2 = self.vega(S=self.SA, K=S0, T=T, r=r, q=q, sigma=sigma)
        self.C3 = self.vega(S=self.SA, K=S0 * 1.1, T=T, r=r, q=q, sigma=sigma)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Underlying Price'
        self.ylabel = 'Vega'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)
    
        
    def vega_vol(self, S0=100, T=0.25, r=0.05, q=0):

        self.sigmaA = np.linspace(0.05, 0.5, 100)
        
        self.C1 = self.vega(S=S0, K=S0 * 0.9, T=T, r=r, q=q, sigma=self.sigmaA)
        self.C2 = self.vega(S=S0, K=S0, T=T, r=r, q=q, sigma=self.sigmaA)
        self.C3 = self.vega(S=S0, K=S0 * 1.1, T=T, r=r, q=q, sigma=self.sigmaA)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Volatility %'
        self.ylabel = 'Vega'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)        
    
    
    def vega_time(self, S0=100, r=0.01, q=0, sigma=0.2):

        self.TA = np.linspace(0.01, 1, 100)
        
        self.C1 = self.vega(S=S0, K=S0 * 0.9, T=self.TA, r=r, q=q, sigma=sigma)
        self.C2 = self.vega(S=S0, K=S0, T=self.TA, r=r, q=q, sigma=sigma)
        self.C3 = self.vega(S=S0, K=S0 * 1.1, T=self.TA, r=r, q=q, sigma=sigma)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Time to Expiration (days)'
        self.ylabel = 'Vega'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)        
    
    
    def theta_price(self, S0=100, T=0.25, r=0.05, q=0, sigma=0.2, option='call'):
        
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)
        
        self.C1 = self.theta(S=self.SA, K=S0 * 0.9, T=T, r=r, q=q, sigma=sigma, option=option)
        self.C2 = self.theta(S=self.SA, K=S0, T=T, r=r, q=q, sigma=sigma, option=option)
        self.C3 = self.theta(S=self.SA, K=S0 * 1.1, T=T, r=r, q=q, sigma=sigma, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Underlying Price'
        self.ylabel = 'Theta'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)        
    
    
    def theta_vol(self, S0=100, T=0.25, r=0.05, q=0, option='call'):
        
        self.sigmaA = np.linspace(0.05, 0.5, 100)
        
        self.C1 = self.theta(S=S0, K=S0 * 0.9, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C2 = self.theta(S=S0, K=S0, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C3 = self.theta(S=S0, K=S0 * 1.1, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Volatility %'
        self.ylabel = 'Delta'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)    
        
    
    def theta_time(self, S0=100, r=0.01, q=0, sigma=0.2, option='call'):
        
        self.TA = np.linspace(0.01, 1, 100)
        
        self.C1 = self.theta(S=S0, K=S0 * 0.9, T=self.TA, r=r, q=q, sigma=sigma, option=option)
        self.C2 = self.theta(S=S0, K=S0, T=self.TA, r=r, q=q, sigma=sigma, option=option)
        self.C3 = self.theta(S=S0, K=S0 * 1.1, T=self.TA, r=r, q=q, sigma=sigma, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Time to Expiration (days)'
        self.ylabel = 'Theta'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel)     
    
    
    def rho_price(self, S0=100, T1=0.25, T2=0.5, r=0.05, q=0, sigma=0.2):
        
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)
        
        self.C1 = self.rho(S=self.SA, K=S0, T=T1, r=r, sigma=sigma, option="call")
        self.C2 = self.rho(S=self.SA, K=S0, T=T2, r=r, sigma=sigma, option="call")
        self.C3 = self.rho(S=self.SA, K=S0, T=T1, r=r, sigma=sigma, option="put")
        self.C4 = self.rho(S=self.SA, K=S0, T=T2, r=r, sigma=sigma, option="put")
        
        self.label1 = '3m call'
        self.label2 = '6m call'
        self.label3 = '3m put'
        self.label4 = '6m put'
                
        self.xlabel = 'Underlying Price'
        self.ylabel = 'Rho'
        
        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             yarray4=self.C4, xarray1=self.SA, xarray2=self.SA, 
                             xarray3=self.SA, xarray4=self.SA, label1=self.label1, 
                             label2=self.label2, label3=self.label3, label4=self.label4, 
                             xlabel=self.xlabel, ylabel=self.ylabel)


    def rho_vol(self, S0=100, T1=0.25, T2=0.5, r=0.05, q=0):
        
        self.sigmaA = np.linspace(0.05, 0.5, 100)
        
        self.C1 = self.rho(S=S0, K=S0, T=T1, r=r, sigma=self.sigmaA, option="call")
        self.C2 = self.rho(S=S0, K=S0, T=T2, r=r, sigma=self.sigmaA, option="call")
        self.C3 = self.rho(S=S0, K=S0, T=T1, r=r, sigma=self.sigmaA, option="put")
        self.C4 = self.rho(S=S0, K=S0, T=T2, r=r, sigma=self.sigmaA, option="put")
        
        self.label1 = '3m call'
        self.label2 = '6m call'
        self.label3 = '3m put'
        self.label4 = '6m put'
                
        self.xlabel = 'Volatility %'
        self.ylabel = 'Rho'
        
        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             yarray4=self.C4, xarray1=self.SA, xarray2=self.SA, 
                             xarray3=self.SA, xarray4=self.SA, label1=self.label1, 
                             label2=self.label2, label3=self.label3, label4=self.label4, 
                             xlabel=self.xlabel, ylabel=self.ylabel)
        
        
    
   
    
    
    def call(self, S0=100, K=100, T=0.25, r=0.05, q=0, sigma=0.2, direction='long', value=False):
        
        self._return_options(legs=1, S0=S0, K1=K, T1=T, r=r, q=q, sigma=sigma, option1='call')
        
        if direction == 'long':
            payoff = self.C1 - self.C1_0
            title = 'Long Call'
            if value == True:
                payoff2 = self.C1_G - self.C1_0
            else:
                payoff2 = None

        if direction == 'short':
            payoff = -self.C1 + self.C1_0
            title = 'Short Call'
            if value == True:
                payoff2 = -self.C1_G + self.C1_0
            else:
                payoff2 = None
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')   
                
        
    def put(self, S0=100, K=100, T=0.25, r=0.05, q=0, sigma=0.2, direction='long', value=False):
        
        self._return_options(legs=1, S0=S0, K1=K, T1=T, r=r, q=q, sigma=sigma, option1='put')
        
        if direction == 'long':
            payoff = self.C1 - self.C1_0
            title = 'Long Put'
            if value == True:
                payoff2 = self.C1_G - self.C1_0
            else:
                payoff2 = None

        if direction == 'short':
            payoff = -self.C1 + self.C1_0
            title = 'Short Put'
            if value == True:
                payoff2 = -self.C1_G + self.C1_0
            else:
                payoff2 = None
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')   
               
        
    def stock(self, S0=100, direction='long'):
        
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)
        
        if direction == 'long':
            payoff = self.SA - S0
            title = 'Long Stock'
        if direction == 'short':
            payoff = S0 - self.SA
            title = 'Short Stock'
        
        self._vis_payoff(S0=S0, SA=self.SA, payoff=payoff, label='Payoff', title=title)     
            
    
    def forward(self, S0=100, K=100, T=0.25, r=0.05, q=0, sigma=0.2, direction='long', cash=False):
        
        self._return_options(legs=2, S0=S0, K1=K, K2=K, T1=T, T2=T, option1='call', option2='put')
        
        if cash == False:
            pv = 1
        else:
            pv = self.discount
        
        if direction == 'long':
            payoff = (self.C1 - self.C2 - self.C1_0 + self.C2_0) * pv
            title = 'Long Forward'
            
        if direction == 'short':
            payoff = -self.C1 + self.C2 + self.C1_0 - self.C2_0 * pv
            title = 'Short Forward'
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, label='Payoff', title=title)
    
    
    def collar(self, S0=50, K1=49, K2=51, T=0.25, r=0.05, q=0, sigma=0.2, direction='long', value=False):
        
        self._return_options(legs=2, S0=S0, K1=K1, K2=K2, T1=T, T2=T, option1='put', 
                             option2='call')
        
        if direction == 'long':
            payoff = self.SA - self.S0 + self.C1 - self.C2 - self.C1_0 + self.C2_0
            title = 'Long Collar'
            if value == True:
                payoff2 = self.SA - self.S0 + self.C1_G - self.C2_G - self.C1_0 + self.C2_0
            else:
                payoff2 = None
                
        if direction == 'short':
            payoff = -self.SA + self.S0 - self.C1 + self.C2 + self.C1_0 - self.C2_0
            title = 'Short Collar'
            if value == True:
                payoff2 = -self.SA + self.S0 - self.C1_G + self.C2_G + self.C1_0 - self.C2_0
            else:
                payoff2 = None
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')

    
    
    def spread(self, S0=100, K1=95, K2=105, T=0.25, r=0.05, q=0, sigma=0.2, option='call', 
               direction='long', value=False):
        
        self._return_options(legs=2, S0=S0, K1=K1, K2=K2, T1=T, T2=T, option1=option, option2=option)
 
        if direction == 'long':        
            payoff = self.C1 - self.C2 - self.C1_0 + self.C2_0
            if value == True:
                payoff2 = self.C1_G - self.C2_G - self.C1_0 + self.C2_0
            else:
                payoff2 = None
                
        if direction == 'short':
            payoff = -self.C1 + self.C2 + self.C1_0 - self.C2_0
            if value == True:
                payoff2 = -self.C1_G + self.C2_G + self.C1_0 - self.C2_0
            else:
                payoff2 = None
                
        if option == 'call' and direction == 'long':
            title = 'Bull Call Spread'
        if option == 'put' and direction == 'long':
            title = 'Bull Put Spread'
        if option == 'call' and direction == 'short':
            title = 'Bear Call Spread'
        if option == 'put' and direction == 'short':
            title = 'Bear Put Spread' 
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
   
    def backspread(self, S0=100, K1=95, K2=105, T=0.25, r=0.05, q=0, sigma=0.2, 
                   option='call', ratio=2, value=False):

        self._return_options(legs=2, S0=S0, K1=K1, K2=K2, T1=T, T2=T, option1=option, option2=option)
        
        if option == 'call':
            title = 'Call Backspread'
            payoff = -self.C1 + (ratio * self.C2) + self.C1_0 - (ratio * self.C2_0)
            if value == True:
                payoff2 = -self.C1_G + (ratio * self.C2_G) + self.C1_0 - (ratio * self.C2_0)
            else:
                payoff2 = None
        
        if option == 'put':
            payoff = ratio * self.C1 - self.C2 - ratio*self.C1_0 + self.C2_0
            title = 'Put Backspread'
            if value == True:
                payoff2 = ratio * self.C1_G - self.C2_G - ratio * self.C1_0 + self.C2_0
            else:
                payoff2 = None
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
        
    def ratio_vertical_spread(self, S0=100, K1=95, K2=105, T=0.25, r=0.05, q=0, 
                              sigma=0.2, option='call', value=False):

        self._return_options(legs=2, S0=S0, K1=K1, K2=K2, T1=T, T2=T, option1=option, option2=option)
        
        if option == 'call':
            title = 'Call Ratio Vertical Spread'
            payoff = self.C1 - 2*self.C2 - self.C1_0 + 2*self.C2_0
            if value == True:
                payoff2 = self.C1_G - 2*self.C2_G - self.C1_0 + 2*self.C2_0
            else:
                payoff2 = None

        if option == 'put':
            title = 'Put Ratio Vertical Spread'
            payoff = -2*self.C1 + self.C2 + 2*self.C1_0 - self.C2_0
            if value == True:
                payoff2 = -2*self.C1_G + self.C2_G + 2*self.C1_0 - self.C2_0
            else:
                payoff2 = None
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
    
    def straddle(self, S0=100, K1=100, K2=100, T=0.25, r=0.05, q=0, sigma=0.2, direction='long', 
                 value=False):
       
        self._return_options(legs=2, S0=S0, K1=K1, K2=K2, T1=T, T2=T, r=r, q=q, 
                             sigma=sigma, option1='call', option2='put')
        
        if direction == 'long':
            payoff = self.C1 + self.C2 - self.C1_0 - self.C2_0
            title = 'Long Straddle'
            if value == True:
                payoff2 = self.C1_G + self.C2_G - self.C1_0 - self.C2_0
            else:
                payoff2 = None
                        
        if direction == 'short':
            payoff = -self.C1 - self.C2 + self.C1_0 + self.C2_0
            title = 'Short Straddle'
            if value == True:
                payoff2 = -self.C1_G - self.C2_G + self.C1_0 + self.C2_0
            else:
                payoff2 = None
            
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, 
                         payoff2=payoff2, label='Payoff', label2='Value')    
  
    
    def strangle(self, S0=100, K1=95, K2=105, T=0.25, r=0.05, q=0, sigma=0.2, direction='long', 
                 value=False):
                
        self._return_options(legs=2, S0=S0, K1=K1, K2=K2, T1=T, T2=T, r=r, q=q, 
                             sigma=sigma, option1='put', option2='call')
        
        if direction == 'long':
            payoff = self.C1 + self.C2 - self.C1_0 - self.C2_0
            title = 'Long Strangle'
            if value == True:
                payoff2 = self.C1_G + self.C2_G - self.C1_0 - self.C2_0
            else:
                payoff2 = None
        
        if direction == 'short':
            payoff = -self.C1 - self.C2 + self.C1_0 + self.C2_0
            title = 'Short Strangle'
            if value == True:
                payoff2 = -self.C1_G - self.C2_G + self.C1_0 + self.C2_0
            else:
                payoff2 = None
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')    


    def butterfly(self, S0=100, K1=95, K2=100, K3=105, T=0.25, r=0.05, q=0, sigma=0.2, 
                  option='call', direction='long', value=False):
        
        self._return_options(legs=3, S0=S0, K1=K1, K2=K2, K3=K3, T1=T, T2=T, T3=T, 
                             r=r, q=q, sigma=sigma, option1=option, option2=option, 
                             option3=option)
        
        if direction == 'long':
            payoff = (self.C1 - 2*self.C2 + self.C3 - self.C1_0 + 2*self.C2_0 - self.C3_0)
            if value == True:
                payoff2 = (self.C1_G - 2*self.C2_G + self.C3_G - self.C1_0 + 2*self.C2_0 - self.C3_0)
            else:
                payoff2 = None
                
        if direction == 'short':    
            payoff = (-self.C1 + 2*self.C2 - self.C3 + self.C1_0 - 2*self.C2_0 + self.C3_0)
            if value == True:
                payoff = (-self.C1_G + 2*self.C2_G - self.C3_G + self.C1_0 - 2*self.C2_0 + self.C3_0)
            else:
                payoff2 = None
    
        if option == 'call' and direction == 'long':
            title = 'Long Butterfly with Calls'
        if option == 'put' and direction == 'long':
            title = 'Long Butterfly with Puts'
        if option == 'call' and direction == 'short':
            title = 'Short Butterfly with Calls'
        if option == 'put' and direction == 'short':
            title = 'Short Butterfly with Puts'
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')


    def christmas_tree(self, S0=100, K1=95, K2=100, K3=105, T=0.25, r=0.05, q=0, 
                       sigma=0.2, option='call', direction='long', value=False):
        
        self._return_options(legs=3, S0=S0, K1=K1, K2=K2, K3=K3, T1=T, T2=T, T3=T, 
                             r=r, q=q, sigma=sigma, option1=option, option2=option, 
                             option3=option)
        
        if option == 'call' and direction == 'long':
            payoff = (self.C1 - self.C2 - self.C3 - self.C1_0 + self.C2_0 + self.C3_0)
            title = 'Long Christmas Tree with Calls'
            if value == True:
                payoff2 = (self.C1_G - self.C2_G - self.C3_G - self.C1_0 + self.C2_0 + self.C3_0)
            else:
                payoff2 = None
                
        if option == 'put' and direction == 'long':
            payoff = (-self.C1 - self.C2 + self.C3 + self.C1_0 + self.C2_0 - self.C3_0)
            title = 'Long Christmas Tree with Puts'
            if value == True:
                payoff2 = (-self.C1_G - self.C2_G + self.C3_G + self.C1_0 + self.C2_0 - self.C3_0)
            else:
                payoff2 = None
            
        if option == 'call' and direction == 'short':
            payoff = (-self.C1 + self.C2 + self.C3 + self.C1_0 - self.C2_0 - self.C3_0)
            title = 'Short Christmas Tree with Calls'
            if value == True:
                payoff2 = (-self.C1_G + self.C2_G + self.C3_G + self.C1_0 - self.C2_0 - self.C3_0)
            else:
                payoff2 = None
            
        if option == 'put' and direction == 'short':
            payoff = (self.C1 + self.C2 - self.C3 - self.C1_0 - self.C2_0 + self.C3_0)
            title = 'Short Christmas Tree with Puts'
            if value == True:
                payoff2 = (self.C1_G + self.C2_G - self.C3_G - self.C1_0 - self.C2_0 + self.C3_0)
            else:
                payoff2 = None
            
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')


    def iron_butterfly(self, S0=100, K1=95, K2=100, K3=100, K4=105, T=0.25, r=0.05, 
                       q=0, sigma=0.2, direction='long', value=False):
        
        self._return_options(legs=4, S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T1=T, T2=T, 
                          T3=T, T4=T, r=r, q=q, sigma=sigma, option1='put', option2='call', 
                          option3='put', option4='call')
        
        if direction == 'long':
            payoff = (-self.C1 + self.C2 + self.C3 - self.C4 + self.C1_0 - 
                      self.C2_0 - self.C3_0 + self.C4_0)
            title = 'Long Iron Butterfly'
            if value == True:
                payoff2 = (-self.C1_G + self.C2_G + self.C3_G - self.C4_G + self.C1_0 - 
                           self.C2_0 - self.C3_0 + self.C4_0)
            else:
                payoff2 = None
        
        if direction == 'short':
            payoff = (self.C1 - self.C2 - self.C3 + self.C4 - self.C1_0 + 
                      self.C2_0 + self.C3_0 - self.C4_0)
            title = 'Short Iron Butterfly'
            if value == True:
                payoff2 = (self.C1_G - self.C2_G - self.C3_G + self.C4_G - self.C1_0 + 
                           self.C2_0 + self.C3_0 - self.C4_0)
            else:
                payoff2 = None
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')

    
    
    def iron_condor(self, S0=100, K1=90, K2=95, K3=100, K4=105, T=0.25, r=0.05, q=0, 
                    sigma=0.2, option='call', direction='long', value=False):
        
        self.return_options(legs=4, S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T1=T, T2=T, 
                          T3=T, T4=T, r=r, q=q, sigma=sigma, option1=option, option2=option, 
                          option3=option, option4=option)
        
        if direction == 'long':
            payoff = (self.C1 - self.C2 - self.C3 + self.C4 - self.C1_0 + 
                      self.C2_0 + self.C3_0 - self.C4_0)
            if value == True:
                payoff2 = (self.C1_G - self.C2_G - self.C3_G + self.C4_G - self.C1_0 + 
                           self.C2_0 + self.C3_0 - self.C4_0)
            else:
                payoff2 = None
        
        if direction == 'short':
            payoff = (-self.C1 + self.C2 + self.C3 - self.C4 + self.C1_0 - 
                      self.C2_0 - self.C3_0 + self.C4_0)
            if value == True:
                payoff2 = (-self.C1_G + self.C2_G + self.C3_G - self.C4_G + self.C1_0 - 
                           self.C2_0 - self.C3_0 + self.C4_0)
            else:
                payoff2 = None
                
        if option == 'call' and direction == 'long':
            title = 'Long Iron Condor with Calls'
        if option == 'put' and direction == 'long':
            title = 'Long Iron Condor with Puts'
        if option == 'call' and direction == 'short':
            title = 'Short Iron Condor with Calls'
        if option == 'put' and direction == 'short':
            title = 'Short Iron Condor with Puts'    
       
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
    
    
    
    def _return_options(self, legs=2, S0=None, K1=None, K2=None, K3=None, K4=None, T1=None, 
                T2=None, T3=None, T4=None, r=None, q=None, sigma=None, option1=None, 
                option2=None, option3=None, option4=None):
 
        self.SA = np.linspace(0.8 * S0, 1.2 * S0, 100)        

        self._initialise_func(S0=S0, SA=self.SA, K1=K1, K2=K2, K3=K3, K4=K4, T1=T1, 
                              T2=T2, T3=T3, T4=T4, r=r, q=q, sigma=sigma, option1=option1, 
                              option2=option2, option3=option3, option4=option4)
        
        self.C1_0 = self.price(S=S0, K=K1, T=T1, r=r, q=q, sigma=sigma, option=option1)
        self.C1 = self.price(S=self.SA, K=K1, T=0, r=r, q=q, sigma=sigma, option=option1)
        self.C1_G = self.price(S=self.SA, K=K1, T=T1, r=r, q=q, sigma=sigma, option=option1)
        
        if legs > 1:
            self.C2_0 = self.price(S=S0, K=K2, T=T2, r=r, q=q, sigma=sigma, option=option2)
            self.C2 = self.price(S=self.SA, K=K2, T=0, r=r, q=q, sigma=sigma, option=option2)
            self.C2_G = self.price(S=self.SA, K=K2, T=T2, r=r, q=q, sigma=sigma, option=option2)

        if legs > 2:
            self.C3_0 = self.price(S=S0, K=K3, T=T3, r=r, q=q, sigma=sigma, option=option3)
            self.C3 = self.price(S=self.SA, K=K3, T=0, r=r, q=q, sigma=sigma, option=option3)
            self.C3_G = self.price(S=self.SA, K=K3, T=T3, r=r, q=q, sigma=sigma, option=option3)
        
        if legs > 3:
            self.C4_0 = self.price(S=S0, K=K4, T=T4, r=r, q=q, sigma=sigma, option=option4)
            self.C4 = self.price(S=self.SA, K=K4, T=0, r=r, q=q, sigma=sigma, option=option4)
            self.C4_G = self.price(S=self.SA, K=K4, T=T4, r=r, q=q, sigma=sigma, option=option4)
        
        return self
        
    
