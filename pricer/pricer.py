import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


# List of default parameters
df_params_list = ['S', 'S0', 'SA', 'K', 'K1', 'K2', 'K3', 'K4', 'H', 'R', 'T', 
                  'T1', 'T2', 'T3', 'T4', 'r', 'b', 'q', 'sigma', 'eta', 'phi', 
                  'barrier_direction', 'knock', 'option', 'option1', 'option2', 
                  'option3', 'option4', 'direction', 'value', 'ratio', 'combo', 
                  'delta_shift', 'delta_shift_type', 'greek', 'interactive', 'notebook', 
                  'colorscheme', 'x_plot', 'y_plot', 'time_shift', 'cash']

mod_payoffs = ['collar', 'straddle', 'butterfly', 'christmas tree',
                       'iron butterfly', 'iron condor']

mod_params = ['S0', 'K1', 'K2', 'K3']

# Dictionary of default parameters
df_dict = {'df_S':100, 
           'df_S0':100,
           'df_SA':np.linspace(80, 120, 100),
           'df_K':100,
           'df_K1':95,
           'df_K2':105,
           'df_K3':105,
           'df_K4':105,
           'df_H':105,
           'df_R':0,
           'df_T':0.25,
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
           'df_barrier_direction':'down',
           'df_knock':'in',    
           'df_option':'call',
           'df_option1':'call',
           'df_option2':'call',
           'df_option3':'call',
           'df_option4':'call',
           'df_direction':'long',
           'df_value':False,
           'df_ratio':2,
           'df_combo':False,
           'df_delta_shift':25,
           'df_delta_shift_type':'avg',
           'df_greek':'delta',
           'df_interactive':False,
           'df_notebook':True,
           'df_colorscheme':'BlueRed',
           'df_x_plot':'delta',
           'df_y_plot':'time',
           'df_time_shift':0.25,
           'df_cash':False}


# Combo parameter values differing from standard defaults
df_combo_dict = {'collar':{'S0':50,
                           'K1':49,
                           'K2':51},
                 'straddle':{'S0':100,
                             'K1':100,
                             'K2':100},
                 'butterfly':{'S0':100,
                              'K1':95,
                              'K2':100,
                              'K3':105},
                 'christmas tree':{'S0':100,
                                   'K1':95,
                                   'K2':100,
                                   'K3':105},
                 'iron butterfly':{'S0':100,
                                   'K1':95,
                                   'K2':100,
                                   'K3':105},
                 'iron condor':{'S0':100,
                                'K1':95,
                                'K2':100,
                                'K3':100}}


class Option():
    
    def __init__(self, S=df_dict['df_S'], S0=df_dict['df_S0'], SA=df_dict['df_SA'], 
                 K=df_dict['df_K'], K1=df_dict['df_K1'], K2=df_dict['df_K2'], K3=df_dict['df_K3'], 
                 K4=df_dict['df_K4'], H=df_dict['df_H'], R=df_dict['df_R'], T=df_dict['df_T'], 
                 T1=df_dict['df_T1'], T2=df_dict['df_T2'], T3=df_dict['df_T3'], 
                 T4=df_dict['df_T4'],r=df_dict['df_r'], b=df_dict['df_b'], 
                 q=df_dict['df_q'], sigma=df_dict['df_sigma'], eta=df_dict['df_eta'], 
                 phi=df_dict['df_phi'], barrier_direction=df_dict['df_barrier_direction'], 
                 knock=df_dict['df_knock'], option=df_dict['df_option'], option1=df_dict['df_option1'], 
                 option2=df_dict['df_option2'], option3=df_dict['df_option3'], 
                 option4=df_dict['df_option4'], direction=df_dict['df_direction'], 
                 value=df_dict['df_value'], ratio=df_dict['df_ratio'], combo=df_dict['df_combo'], 
                 delta_shift=df_dict['df_delta_shift'],
                 delta_shift_type=df_dict['df_delta_shift_type'], greek=df_dict['df_greek'], 
                 interactive=df_dict['df_interactive'], notebook=df_dict['df_notebook'], 
                 colorscheme=df_dict['df_colorscheme'], x_plot=df_dict['df_x_plot'], 
                 y_plot=df_dict['df_y_plot'], time_shift=df_dict['df_time_shift'], 
                 cash=df_dict['df_cash'], df_combo_dict=df_combo_dict, df_dict=df_dict, 
                 df_params_list=df_params_list, mod_payoffs=mod_payoffs, mod_params=mod_params):

        self.S = S # Spot price
        self.S0 = S0 # Spot price
        self.SA = SA # Array of spot prices
        self.K = K # Strike price
        self.K1 = K1 # Strike price
        self.K2 = K2 # Strike price
        self.K3 = K3 # Strike price
        self.K4 = K4 # Strike price
        self.H = H # Barrier level
        self.R = R # Rebate
        self.T = T # Time to maturity
        self.T1 = T1 # Time to maturity
        self.T2 = T2 # Time to maturity
        self.T3 = T3 # Time to maturity
        self.T4 = T4 # Time to maturity
        self.r = r # Interest rate
        self.q = q # Dividend Yield 
        self.b = self.r - self.q # Cost of carry
        self.sigma = sigma # Volatility
        self.eta = eta # Barrier parameter
        self.phi = phi # Barrier parameter
        self.barrier_direction = barrier_direction # Whether strike is up or down
        self.knock = knock # Whether option knocks in or out
        self.option = option # Option type, call or put
        self.option1 = option1 # Option type, call or put
        self.option2 = option2 # Option type, call or put
        self.option3 = option3 # Option type, call or put
        self.option4 = option4 # Option type, call or put
        self.direction = direction # Payoff direction, long or short
        self.value = value # Flag whether to plot Intrinsic Value against payoff
        self.ratio = ratio # Ratio used in Backspread and Ratio Vertical Spread 
        self.combo = combo # Flag whether to refresh default values in price formula
        self.delta_shift = delta_shift # Size of shift used in shift_delta function
        self.delta_shift_type = delta_shift_type # Shift type - Up, Down or Avg
        self.df_dict = df_dict # Dictionary of parameter defaults
        self.df_combo_dict = df_combo_dict # Dictionary of payoffs with different default parameters
        self.df_params_list = df_params_list # List of default parameters
        self.greek = greek # Option greek to display e.g. delta
        self.interactive = interactive # Whether to display static mpl 3D graph or plotly interactive graph
        self.notebook = notebook # Whether running in iPython notebook or not, False creates a popup html page 
        self.colorscheme = colorscheme # Color palette to use in 3D graphs
        self.x_plot = x_plot # X-axis in 2D greeks graph
        self.y_plot = y_plot # Y-axis in 2D greeks graph
        self.time_shift = time_shift # Time between periods used in 2D greeks graph
        self.cash = cash # Whether to graph forward at cash or discount
        self.mod_payoffs = mod_payoffs # Combo payoffs needing different default parameters
        self.mod_params = mod_params # Parameters of these payoffs that need changing
        self.combo_payoff = None

    
    def _initialise_func(self, **kwargs):
        
        self._refresh_params(**kwargs)
        self._refresh_dist()
        
        return self

    def _initialise_combo(self, **kwargs):
        self._set_params(**kwargs)
        self._refresh_dist()
        
        return self
    

    def _initialise_barriers(self, **kwargs):
        self._refresh_params(**kwargs)
        self._refresh_dist()
        self._barrier_factors()

        return self


    def _refresh_params(self, **kwargs):

        if self.combo_payoff in self.mod_payoffs:
            for k, v in kwargs.items():
                if v is None:
                    if k in self.mod_params:
                        try:
                            v = self.df_combo_dict[str(self.combo_payoff)][str(k)]
                        except:
                            v = df_dict['df_'+str(k)]
                    if k not in self.mod_params:
                        v = df_dict['df_'+str(k)]
                    self.__dict__[k] = v
                else:
                    self.__dict__[k] = v
        
            
        else:
            for k, v in kwargs.items():
                if v is None:
                    v = df_dict['df_'+str(k)]
                    self.__dict__[k] = v
                else:
                    self.__dict__[k] = v

        for key in list(set(self.df_params_list) - set(self.mod_params)):
            if key not in kwargs:
                val = df_dict['df_'+str(key)]
                self.__dict__[key] = val
                
        return self        
   
    
    def _set_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v
    
        return self
       
        
    def _refresh_dist(self):
        
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


    def price(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              combo=None):
        """
        Return the Black Scholes Option Price

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100, taken from dictionary of default parameters. 
        K : Float
            Strike Price. The default is 100, taken from dictionary of default parameters.
        T : Float
            Time to Maturity. The default is 0.5 (6 months), taken from dictionary of default parameters.
        r : Float
            Interest Rate. The default is 0.05 (5%), taken from dictionary of default parameters.
        q : Float
            Dividend Yield. The default is 0, taken from dictionary of default parameters.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%), taken from dictionary of default parameters.
        option : Str
            Option type, Put or Call. The default is 'call'

        Returns
        -------
        Float
            Black Scholes Option Price. If combo is set to true the price to be used 
            in combo graphs so the distributions are refreshed but not the parameters.

        """
        if combo == False or combo is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if combo == True:
            self._initialise_combo(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        
        if self.option == "call":
            self.opt_price = ((self.S * self.carry * self.Nd1) - 
                              (self.K * np.exp(-self.r * self.T) * self.Nd2))  
        if self.option == 'put':
            self.opt_price = ((self.K * np.exp(-self.r * self.T) * self.minusNd2) - 
                              (self.S * self.carry * self.minusNd1))
        
        np.nan_to_num(self.opt_price, copy=False)
                
        return self.opt_price


    def delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None):
                    
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
                        
        if self.option == 'call':
            self.opt_delta = self.carry * self.Nd1
        if self.option == 'put':
            self.opt_delta = -self.carry * self.minusNd1
            
        return self.opt_delta
    
    
    def shift_delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
                    shift=None, shift_type=None):
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                              shift=shift, shift_type=shift_type)
        
        down_shift = self.S-(self.shift/10000)*self.S
        up_shift = self.S+(self.shift/10000)*self.S
        opt_price = self.price(S=self.S, K=self.K, T=self.T, r=self.r, q=self.q, 
                               sigma=self.sigma, option=self.option)
        op_shift_down = self.price(S=down_shift, K=self.K, T=self.T, r=self.r, 
                                   q=self.q, sigma=self.sigma, option=self.option)
        op_shift_up = self.price(S=up_shift, K=self.K, T=self.T, r=self.r, q=self.q, 
                                 sigma=self.sigma, option=self.option)
                
        if self.shift_type == 'up':
            self.opt_delta_shift = (op_shift_up - opt_price) * 4
        if self.shift_type == 'down':
            self.opt_delta_shift = (opt_price - op_shift_down) * 4
        if self.shift_type == 'avg':    
            self.opt_delta_shift = (op_shift_up - op_shift_down) * 2
        
        return self.opt_delta_shift
    
    
    def theta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None):
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
    
        if self.option == 'call':
            self.opt_theta = ((-self.S * self.carry * self.nd1 * self.sigma ) / 
                              (2 * np.sqrt(self.T)) - (self.b - self.r) * self.S * self.carry * 
                              self.Nd1 - self.r * self.K * np.exp(-self.r * self.T) * self.Nd2)
        if self.option == 'put':   
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
    
    
    def rho(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None):
                
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        
        if self.option == 'call':
            self.opt_rho = self.T * self.K * np.exp(-self.r * self.T) * self.Nd2
        if self.option == 'put':
            self.opt_rho = -self.T * self.K * np.exp(-self.r * self.T) * self.minusNd2
            
        return self.opt_rho


    def vanna(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        # aka DdeltaDvol, DvegaDspot 
        # how much delta will change due to a small change in volatility
        # how much vega will change due to a small change in the asset price        
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        
        self.opt_vanna = ((-self.carry * self.d2) / self.sigma) * self.nd1 

        return self.opt_vanna               
           

    def charm(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None):
        # aka DdeltaDtime, Delta Bleed 
        # how much delta will change due to a small change in time        
        
        self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        
        if self.option == 'call':
            self.opt_charm = (-self.carry * ((self.nd1 * ((self.b / (self.sigma * np.sqrt(self.T))) - 
                                                          (self.d2 / (2 * self.T)))) + 
                                             ((self.b - self.r) * self.Nd1)))
        if self.option == 'put':
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
                       sigma=None, barrier_direction=None, knock=None, option=None):
        
        self._initialise_barriers(S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                                  barrier_direction=barrier_direction, knock=knock, 
                                  option=option)
        
        if self.barrier_direction == 'down' and self.knock == 'in' and self.option == 'call':
            self.eta = 1
            self.phi = 1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.C + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.B + self.D + self.E
            

        if self.barrier_direction == 'up' and self.knock == 'in' and self.option == 'call':
            self.eta = -1
            self.phi = 1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.A + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.B - self.C + self.D + self.E


        if self.barrier_direction == 'down' and self.knock == 'in' and self.option == 'put':
            self.eta = 1
            self.phi = -1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.B - self.C + self.D + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.A + self.E
                
         
        if self.barrier_direction == 'up' and self.knock == 'in' and self.option == 'put':
            self.eta = -1
            self.phi = -1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.B + self.D + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.C + self.E
                

        if self.barrier_direction == 'down' and self.knock == 'out' and self.option == 'call':
            self.eta = 1
            self.phi = 1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.C + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.B - self.D + self.F
            

        if self.barrier_direction == 'up' and self.knock == 'out' and self.option == 'call':
            self.eta = -1
            self.phi = 1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.B + self.C - self.D + self.F


        if self.barrier_direction == 'down' and self.knock == 'out' and self.option == 'put':
            self.eta = 1
            self.phi = -1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.B + self.C - self.D + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.F
                
         
        if self.barrier_direction == 'up' and self.knock == 'out' and self.option == 'put':
            self.eta = -1
            self.phi = -1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.B - self.D + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.C + self.F

        return self.opt_barrier_payoff    



    def _vis_payoff(self, S0=None, SA=None, payoff=None, label=None, title='Option Payoff', 
                    payoff2=None, label2=None, payoff3=None, label3=None, payoff4=None, 
                    label4=None, xlabel='Stock Price', ylabel='P&L'):
       
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
    
    
    def _vis_greeks_mpl(self, S0=None, xarray1=None, xarray2=None, xarray3=None, 
                        xarray4=None, yarray1=None, yarray2=None, yarray3=None, 
                        yarray4=None, label1=None, label2=None, label3=None, label4=None, 
                        xlabel=None, ylabel=None, title='Payoff'):
        
        fig, ax = plt.subplots()
        ax.plot(xarray1, yarray1, color='blue', label=label1)
        ax.plot(xarray2, yarray2, color='red', label=label2)
        ax.plot(xarray3, yarray3, color='green', label=label3)
        if label4 is not None:
            ax.plot(xarray4, yarray4, color='orange', label=label4)
        plt.grid(True)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend()
        plt.show()
   
    
    def greeks_graphs_3D(self, S0=None, r=None, q=None, sigma=None, greek=None, option=None, 
                         interactive=None, notebook=None, colorscheme=None):

        self._initialise_func(S0=S0, r=r, q=q, sigma=sigma, greek=greek, option=option, 
                         interactive=interactive, notebook=notebook, colorscheme=colorscheme)

        self.TA_lower = 0.01

        if greek == 'price':
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 1
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.price(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma, 
                                option=self.option)

        if greek == 'delta':
            self.SA_lower = 0.25
            self.SA_upper = 1.75
            self.TA_upper = 2
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.delta(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma, 
                                option=self.option)

        if greek == 'gamma':
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.gamma(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)

        if greek == 'vega':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vega(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)

        if greek == 'theta':    
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 1
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.theta(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma, 
                                option=self.option)
            
        if greek == 'rho':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.rho(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma, 
                              option=self.option)    

        if greek == 'vomma':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vomma(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)

        if greek == 'vanna':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vanna(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)

        if greek == 'zomma':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.zomma(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)
            
        if greek == 'speed':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.speed(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)    

        if greek == 'color':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.5
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.color(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma) 
            
        if greek == 'ultima':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.ultima(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)     

        if greek == 'vega bleed':               
            self.SA_lower = 0.5
            self.SA_upper = 1.5
            self.TA_upper = 1
            self.option = 'Call / Put'
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.vega_bleed(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma)   

        if greek == 'charm':               
            self.SA_lower = 0.8
            self.SA_upper = 1.2
            self.TA_upper = 0.25
            self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
            self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.z = self.charm(S=self.x, K=self.S0, T=self.y, r=self.r, sigma=self.sigma, 
                                option=self.option)

        if self.option == 'Call / Put':
            titlename = str(self.option+' Option '+str(self.greek.title()))
        else:    
            titlename = str(str(self.option.title())+' Option '+str(self.greek.title()))


        if self.interactive == False:
        
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
            ax.set_zlabel(str(self.greek.title()), fontsize=12)
            ax.set_title(titlename, fontsize=14)
            plt.show()


        if self.interactive == True:
            
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
                                             colorscale=self.colorscheme, 
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
                                zaxis_title=str(self.greek.title()),),
                              title=titlename, autosize=False, 
                              width=800, height=800,
                              margin=dict(l=65, r=50, b=65, t=90),
                             scene_camera=camera)
            
            if self.notebook == True:
                fig.show()
            else:
                plot(fig, auto_open=True)
 
    
    def greeks_graphs_2D(self, x_plot=None, y_plot=None, S0=None, T=None, time_shift=None,
                         r=None, q=None, sigma=None, option=None):
        
        self._initialise_func(x_plot=x_plot, y_plot=y_plot, S0=S0, T=T, time_shift=time_shift, 
                              r=r, q=q, sigma=sigma, option=option)
        
        if self.x_plot == 'value':
            if self.y_plot == 'price':
                self.value_price(S0=self.S0, T=self.T, r=self.r, q=self.q, sigma=self.sigma, 
                                 option=self.option)
            if self.y_plot == 'vol':
                self.value_vol(S0=self.S0, T=self.T, r=self.r, q=self.q, option=option)
            if self.y_plot == 'time':
                self.value_time(S0=self.S0, r=self.r, q=self.q, sigma=self.sigma, 
                                option=self.option)
        
        if self.x_plot == 'delta':
            if self.y_plot == 'price':
                self.delta_price(S0=self.S0, T=self.T, r=self.r, q=self.q, sigma=self.sigma, 
                                 option=self.option)
            if self.y_plot == 'vol':
                self.delta_vol(S0=self.S0, T=self.T, r=self.r, q=self.q, option=option)
            if self.y_plot == 'time':
                self.delta_time(S0=self.S0, r=self.r, q=self.q, sigma=self.sigma, 
                                option=self.option)
        
        if self.x_plot == 'gamma':
            if self.y_plot == 'price':
                self.gamma_price(S0=self.S0, T=self.T, r=self.r, q=self.q, sigma=self.sigma)
            if self.y_plot == 'vol':
                self.gamma_vol(S0=self.S0, T=self.T, r=self.r, q=self.q)
            if self.y_plot == 'time':
                self.gamma_time(S0=self.S0, r=self.r, q=self.q, sigma=self.sigma)
        
        if self.x_plot == 'vega':
            if self.y_plot == 'price':
                self.vega_price(S0=self.S0, T=self.T, r=self.r, q=self.q, sigma=self.sigma)
            if self.y_plot == 'vol':
                self.vega_vol(S0=self.S0, T=self.T, r=self.r, q=self.q)
            if self.y_plot == 'time':
                self.vega_time(S0=self.S0, r=self.r, q=self.q, sigma=self.sigma)  
        
        if self.x_plot == 'theta':
            if self.y_plot == 'price':
                self.theta_price(S0=self.S0, T=self.T, r=self.r, q=self.q, sigma=self.sigma, 
                                 option=self.option)
            if self.y_plot == 'vol':
                self.theta_vol(S0=self.S0, T=self.T, r=self.r, q=self.q, option=option)
            if self.y_plot == 'time':
                self.theta_time(S0=self.S0, r=self.r, q=self.q, sigma=self.sigma, 
                                option=self.option)
        
        if self.x_plot == 'rho':
            if self.y_plot == 'price':
                self.rho_price(S0=self.S0, T1=self.T, T2=self.T+self.time_shift, 
                               r=self.r, q=self.q, sigma=self.sigma)
            if self.y_plot == 'vol':
                self.rho_vol(S0=self.S0, T1=self.T, T2=self.T+self.time_shift, 
                               r=self.r, q=self.q)
    

        
    def payoff_graphs(self, S0=None, K=None, K1=None, K2=None, K3=None, K4=None, 
                      T=None, r=None, q=None, sigma=None, option=None, direction=None, 
                      cash=None, value=None, combo_payoff=None):
       
        if combo_payoff == 'call':
            self.call(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                      value=value)
        
        if combo_payoff == 'put':
            self.put(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                      value=value)
        
        if combo_payoff == 'stock':
            self.stock(S0=S0, direction=direction)
        
        if combo_payoff == 'forward':
            self.forward(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction,
                         cash=cash)
        
        if combo_payoff == 'collar':
            self.collar(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                        value=value)
        
        if combo_payoff == 'spread':
            self.spread(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                        direction=direction, value=value)
            
        if combo_payoff == 'backspread':
            self.backspread(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                            value=value)
        
        if combo_payoff == 'ratio vertical spread':
            self.ratio_vertical_spread(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                                       option=option, value=value)
        
        if combo_payoff == 'straddle':
            self.straddle(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                          value=value)

        if combo_payoff == 'strangle':
            self.strangle(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                          value=value)
        
        if combo_payoff == 'butterfly':    
            self.butterfly(S0=S0, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                           option=option, direction=direction, value=value)
        
        if combo_payoff == 'christmas tree':
            self.christmas_tree(S0=S0, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                                option=option, direction=direction, value=value)    
        
        if combo_payoff == 'iron butterfly':
            self.iron_butterfly(S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                                sigma=sigma, direction=direction, value=value)
            
        if combo_payoff == 'iron condor':
            self.iron_condor(S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                             sigma=sigma, option=option, direction=direction, value=value)
            
    
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
        self.title = 'Value vs Price'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)        
    
    
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
        self.title = 'Value vs Volatility'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)     
        
    
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
        self.title = 'Value vs Time'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)     
    
    
    
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
        self.title = 'Delta vs Price'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)       
    
    
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
        self.title = 'Delta vs Volatility'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)        
        
        
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
        self.title = 'Delta vs Time'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)     
    
        
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
        self.title = 'Gamma vs Price'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)        
        
        
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
        self.title = 'Gamma vs Volatility'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)            
        
    
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
        self.title = 'Gamma vs Time'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)     
    
    
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
        self.title = 'Vega vs Price'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
    
        
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
        self.title = 'Vega vs Volatility'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)        
    
    
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
        self.title = 'Vega vs Time'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)        
    
    
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
        self.title = 'Theta vs Price'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.SA, xarray2=self.SA, xarray3=self.SA, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)        
    
    
    def theta_vol(self, S0=100, T=0.25, r=0.05, q=0, option='call'):
        
        self.sigmaA = np.linspace(0.05, 0.5, 100)
        
        self.C1 = self.theta(S=S0, K=S0 * 0.9, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C2 = self.theta(S=S0, K=S0, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
        self.C3 = self.theta(S=S0, K=S0 * 1.1, T=T, r=r, q=q, sigma=self.sigmaA, option=option)
    
        self.label1 = str(int(S0 * 0.9))+' Strike'
        self.label2 = 'ATM Strike'
        self.label3 = str(int(S0 * 1.1))+' Strike'
            
        self.xlabel = 'Volatility %'
        self.ylabel = 'Theta'
        self.title = 'Theta vs Volatility'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.sigmaA*100, xarray2=self.sigmaA*100, xarray3=self.sigmaA*100, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)    
        
    
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
        self.title = 'Theta vs Time'

        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             xarray1=self.TA*365, xarray2=self.TA*365, xarray3=self.TA*365, 
                             label1=self.label1, label2=self.label2, label3=self.label3, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)     
    
    
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
        self.title = 'Rho vs Price'
        
        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             yarray4=self.C4, xarray1=self.SA, xarray2=self.SA, 
                             xarray3=self.SA, xarray4=self.SA, label1=self.label1, 
                             label2=self.label2, label3=self.label3, label4=self.label4, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)


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
        self.title = 'Rho vs Volatility'
        
        self._vis_greeks_mpl(S0=self.S0, yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                             yarray4=self.C4, xarray1=self.SA, xarray2=self.SA, 
                             xarray3=self.SA, xarray4=self.SA, label1=self.label1, 
                             label2=self.label2, label3=self.label3, label4=self.label4, 
                             xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)

    
    def call(self, S0=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, value=None):
        
        self.combo_payoff = 'call'
        
        self._initialise_func(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, value=value)
        
        self._return_options(legs=1, S0=self.S0, K1=self.K, T1=self.T, r=self.r, 
                             q=self.q, sigma=self.sigma, option1='call')
        
        if self.direction == 'long':
            payoff = self.C1 - self.C1_0
            title = 'Long Call'
            if self.value == True:
                payoff2 = self.C1_G - self.C1_0
            else:
                payoff2 = None

        if self.direction == 'short':
            payoff = -self.C1 + self.C1_0
            title = 'Short Call'
            if self.value == True:
                payoff2 = -self.C1_G + self.C1_0
            else:
                payoff2 = None
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')   
                
        
    def put(self, S0=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, value=None):
        
        self.combo_payoff = 'put'
        
        self._initialise_func(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, value=value)
        
        self._return_options(legs=1, S0=self.S0, K1=self.K, T1=self.T, r=self.r, 
                             q=self.q, sigma=self.sigma, option1='put')
        
        if self.direction == 'long':
            payoff = self.C1 - self.C1_0
            title = 'Long Put'
            if self.value == True:
                payoff2 = self.C1_G - self.C1_0
            else:
                payoff2 = None

        if self.direction == 'short':
            payoff = -self.C1 + self.C1_0
            title = 'Short Put'
            if self.value == True:
                payoff2 = -self.C1_G + self.C1_0
            else:
                payoff2 = None
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')   
               
        
    def stock(self, S0=None, direction=None):
        
        self.combo_payoff = 'stock'
        
        self._initialise_func(S0=S0, direction=direction)
        
        self.SA = np.linspace(0.8 * self.S0, 1.2 * self.S0, 100)
        
        if self.direction == 'long':
            payoff = self.SA - self.S0
            title = 'Long Stock'
        if self.direction == 'short':
            payoff = self.S0 - self.SA
            title = 'Short Stock'
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, label='Payoff', title=title)     
            
    
    def forward(self, S0=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, 
                cash=None):
        
        self.combo_payoff = 'forward'
        
        self._initialise_func(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, cash=cash)
        
        self._return_options(legs=2, S0=self.S0, K1=self.K, K2=self.K, T1=self.T, 
                             T2=self.T, option1='call', option2='put')
        
        if self.cash == False:
            pv = 1
        else:
            pv = self.discount
        
        if self.direction == 'long':
            payoff = (self.C1 - self.C2 - self.C1_0 + self.C2_0) * pv
            title = 'Long Forward'
            
        if self.direction == 'short':
            payoff = -self.C1 + self.C2 + self.C1_0 - self.C2_0 * pv
            title = 'Short Forward'
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, label='Payoff', title=title)
    
    
    def collar(self, S0=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
               direction=None, value=None):
        
        self.combo_payoff = 'collar'
        
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                              direction=direction, value=value)

        self._return_options(legs=2, S0=self.S0, K1=self.K1, K2=self.K2, T1=self.T, 
                             T2=self.T, option1='put', option2='call')
        
        if self.direction == 'long':
            payoff = self.SA - self.S0 + self.C1 - self.C2 - self.C1_0 + self.C2_0
            title = 'Long Collar'
            if self.value == True:
                payoff2 = self.SA - self.S0 + self.C1_G - self.C2_G - self.C1_0 + self.C2_0
            else:
                payoff2 = None
                
        if self.direction == 'short':
            payoff = -self.SA + self.S0 - self.C1 + self.C2 + self.C1_0 - self.C2_0
            title = 'Short Collar'
            if self.value == True:
                payoff2 = -self.SA + self.S0 - self.C1_G + self.C2_G + self.C1_0 - self.C2_0
            else:
                payoff2 = None
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')

    
    
    def spread(self, S0=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
               option=None, direction=None, value=None):
        
        self.combo_payoff = 'spread'
        
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                              direction=direction, value=value)
        
        self._return_options(legs=2, S0=self.S0, K1=self.K1, K2=self.K2, T1=self.T, 
                             T2=self.T, option1=self.option, option2=self.option)
 
        if self.direction == 'long':        
            payoff = self.C1 - self.C2 - self.C1_0 + self.C2_0
            if self.value == True:
                payoff2 = self.C1_G - self.C2_G - self.C1_0 + self.C2_0
            else:
                payoff2 = None
                
        if self.direction == 'short':
            payoff = -self.C1 + self.C2 + self.C1_0 - self.C2_0
            if self.value == True:
                payoff2 = -self.C1_G + self.C2_G + self.C1_0 - self.C2_0
            else:
                payoff2 = None
                
        if self.option == 'call' and self.direction == 'long':
            title = 'Bull Call Spread'
        if self.option == 'put' and self.direction == 'long':
            title = 'Bull Put Spread'
        if self.option == 'call' and self.direction == 'short':
            title = 'Bear Call Spread'
        if self.option == 'put' and self.direction == 'short':
            title = 'Bear Put Spread' 
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
   
    def backspread(self, S0=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
                   option=None, ratio=None, value=None):

        self.combo_payoff = 'backspread'

        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                              ratio=ratio, value=value)
        
        self._return_options(legs=2, S0=self.S0, K1=self.K1, K2=self.K2, T1=self.T, 
                             T2=self.T, option1=self.option, option2=self.option)
        
        if self.option == 'call':
            title = 'Call Backspread'
            payoff = -self.C1 + (self.ratio * self.C2) + self.C1_0 - (self.ratio * self.C2_0)
            if self.value == True:
                payoff2 = -self.C1_G + (self.ratio * self.C2_G) + self.C1_0 - (self.ratio * self.C2_0)
            else:
                payoff2 = None
        
        if self.option == 'put':
            payoff = self.ratio * self.C1 - self.C2 - self.ratio * self.C1_0 + self.C2_0
            title = 'Put Backspread'
            if self.value == True:
                payoff2 = self.ratio * self.C1_G - self.C2_G - self.ratio * self.C1_0 + self.C2_0
            else:
                payoff2 = None
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
        
    def ratio_vertical_spread(self, S0=None, K1=None, K2=None, T=None, r=None, q=None, 
                              sigma=None, option=None, ratio=None, value=None):

        self.combo_payoff = 'ratio vertical spread'

        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                              ratio=ratio, value=value)
        
        self._return_options(legs=2, S0=self.S0, K1=self.K1, K2=self.K2, T1=self.T, 
                             T2=self.T, option1=self.option, option2=self.option)
        
        if self.option == 'call':
            title = 'Call Ratio Vertical Spread'
            payoff = self.C1 - self.ratio * self.C2 - self.C1_0 + self.ratio * self.C2_0
            if self.value == True:
                payoff2 = self.C1_G - self.ratio * self.C2_G - self.C1_0 + self.ratio * self.C2_0
            else:
                payoff2 = None

        if self.option == 'put':
            title = 'Put Ratio Vertical Spread'
            payoff = -self.ratio * self.C1 + self.C2 + self.ratio * self.C1_0 - self.C2_0
            if self.value == True:
                payoff2 = -self.ratio * self.C1_G + self.C2_G + self.ratio * self.C1_0 - self.C2_0
            else:
                payoff2 = None
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
    
    def straddle(self, S0=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, 
                 value=None):
       
        self.combo_payoff = 'straddle'
        
        self._initialise_func(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                              value=value)
                
        self._return_options(legs=2, S0=self.S0, K1=self.K, K2=self.K, T1=self.T, 
                             T2=self.T, r=self.r, q=self.q, sigma=self.sigma, 
                             option1='call', option2='put')
        
        if self.direction == 'long':
            payoff = self.C1 + self.C2 - self.C1_0 - self.C2_0
            title = 'Long Straddle'
            if self.value == True:
                payoff2 = self.C1_G + self.C2_G - self.C1_0 - self.C2_0
            else:
                payoff2 = None
                        
        if self.direction == 'short':
            payoff = -self.C1 - self.C2 + self.C1_0 + self.C2_0
            title = 'Short Straddle'
            if self.value == True:
                payoff2 = -self.C1_G - self.C2_G + self.C1_0 + self.C2_0
            else:
                payoff2 = None
            
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, 
                         payoff2=payoff2, label='Payoff', label2='Value')    
  
    
    def strangle(self, S0=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
                 direction=None, value=None):
        
        self.combo_payoff = 'strangle'
        
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                              value=value)
        
        self._return_options(legs=2, S0=self.S0, K1=self.K1, K2=self.K2, T1=self.T, 
                             T2=self.T, r=self.r, q=self.q, sigma=self.sigma, option1='put', 
                             option2='call')
        
        if self.direction == 'long':
            payoff = self.C1 + self.C2 - self.C1_0 - self.C2_0
            title = 'Long Strangle'
            if self.value == True:
                payoff2 = self.C1_G + self.C2_G - self.C1_0 - self.C2_0
            else:
                payoff2 = None
        
        if self.direction == 'short':
            payoff = -self.C1 - self.C2 + self.C1_0 + self.C2_0
            title = 'Short Strangle'
            if self.value == True:
                payoff2 = -self.C1_G - self.C2_G + self.C1_0 + self.C2_0
            else:
                payoff2 = None
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')    


    def butterfly(self, S0=None, K1=None, K2=None, K3=None, T=None, r=None, q=None, 
                  sigma=None, option=None, direction=None, value=None):
        
        self.combo_payoff = 'butterfly'
        
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                              option=option, direction=direction, value=value)
        
        self._return_options(legs=3, S0=self.S0, K1=self.K1, K2=self.K2, K3=self.K3, 
                             T1=self.T, T2=self.T, T3=self.T, r=self.r, q=self.q, 
                             sigma=self.sigma, option1=self.option, option2=self.option, 
                             option3=self.option)
        
        if self.direction == 'long':
            payoff = (self.C1 - 2*self.C2 + self.C3 - self.C1_0 + 2*self.C2_0 - self.C3_0)
            if self.value == True:
                payoff2 = (self.C1_G - 2*self.C2_G + self.C3_G - self.C1_0 + 2*self.C2_0 - self.C3_0)
            else:
                payoff2 = None
                
        if self.direction == 'short':    
            payoff = (-self.C1 + 2*self.C2 - self.C3 + self.C1_0 - 2*self.C2_0 + self.C3_0)
            if self.value == True:
                payoff = (-self.C1_G + 2*self.C2_G - self.C3_G + self.C1_0 - 2*self.C2_0 + self.C3_0)
            else:
                payoff2 = None
    
        if self.option == 'call' and self.direction == 'long':
            title = 'Long Butterfly with Calls'
        if self.option == 'put' and self.direction == 'long':
            title = 'Long Butterfly with Puts'
        if self.option == 'call' and self.direction == 'short':
            title = 'Short Butterfly with Calls'
        if self.option == 'put' and self.direction == 'short':
            title = 'Short Butterfly with Puts'
                
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')


    def christmas_tree(self, S0=None, K1=None, K2=None, K3=None, T=None, r=None, 
                       q=None, sigma=None, option=None, direction=None, value=None):
        
        self.combo_payoff = 'christmas tree'
        
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                              option=option, direction=direction, value=value)
        
        self._return_options(legs=3, S0=self.S0, K1=self.K1, K2=self.K2, K3=self.K3, 
                             T1=self.T, T2=self.T, T3=self.T, r=self.r, q=self.q, 
                             sigma=self.sigma, option1=self.option, option2=self.option, 
                             option3=self.option)
        
        if self.option == 'call' and self.direction == 'long':
            payoff = (self.C1 - self.C2 - self.C3 - self.C1_0 + self.C2_0 + self.C3_0)
            title = 'Long Christmas Tree with Calls'
            if self.value == True:
                payoff2 = (self.C1_G - self.C2_G - self.C3_G - self.C1_0 + self.C2_0 + self.C3_0)
            else:
                payoff2 = None
                
        if self.option == 'put' and self.direction == 'long':
            payoff = (-self.C1 - self.C2 + self.C3 + self.C1_0 + self.C2_0 - self.C3_0)
            title = 'Long Christmas Tree with Puts'
            if self.value == True:
                payoff2 = (-self.C1_G - self.C2_G + self.C3_G + self.C1_0 + self.C2_0 - self.C3_0)
            else:
                payoff2 = None
            
        if self.option == 'call' and self.direction == 'short':
            payoff = (-self.C1 + self.C2 + self.C3 + self.C1_0 - self.C2_0 - self.C3_0)
            title = 'Short Christmas Tree with Calls'
            if self.value == True:
                payoff2 = (-self.C1_G + self.C2_G + self.C3_G + self.C1_0 - self.C2_0 - self.C3_0)
            else:
                payoff2 = None
            
        if self.option == 'put' and self.direction == 'short':
            payoff = (self.C1 + self.C2 - self.C3 - self.C1_0 - self.C2_0 + self.C3_0)
            title = 'Short Christmas Tree with Puts'
            if self.value == True:
                payoff2 = (self.C1_G + self.C2_G - self.C3_G - self.C1_0 - self.C2_0 + self.C3_0)
            else:
                payoff2 = None
            
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')


    def iron_butterfly(self, S0=None, K1=None, K2=None, K3=None, K4=None, T=None, r=None, 
                       q=None, sigma=None, direction=None, value=None):
        
        self.combo_payoff = 'iron butterfly'
        
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                              sigma=sigma, direction=direction, value=value)
        
        self._return_options(legs=4, S0=self.S0, K1=self.K1, K2=self.K2, K3=self.K3, 
                             K4=self.K4, T1=self.T, T2=self.T, T3=self.T, T4=self.T, 
                             r=self.r, q=self.q, sigma=self.sigma, option1='put', option2='call', 
                             option3='put', option4='call')
        
        if self.direction == 'long':
            payoff = (-self.C1 + self.C2 + self.C3 - self.C4 + self.C1_0 - 
                      self.C2_0 - self.C3_0 + self.C4_0)
            title = 'Long Iron Butterfly'
            if self.value == True:
                payoff2 = (-self.C1_G + self.C2_G + self.C3_G - self.C4_G + self.C1_0 - 
                           self.C2_0 - self.C3_0 + self.C4_0)
            else:
                payoff2 = None
        
        if self.direction == 'short':
            payoff = (self.C1 - self.C2 - self.C3 + self.C4 - self.C1_0 + 
                      self.C2_0 + self.C3_0 - self.C4_0)
            title = 'Short Iron Butterfly'
            if self.value == True:
                payoff2 = (self.C1_G - self.C2_G - self.C3_G + self.C4_G - self.C1_0 + 
                           self.C2_0 + self.C3_0 - self.C4_0)
            else:
                payoff2 = None
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
    
    
    def iron_condor(self, S0=None, K1=None, K2=None, K3=None, K4=None, T=None, r=None, 
                       q=None, sigma=None, option=None, direction=None, value=None):
        
        self.combo_payoff = 'iron condor'
        
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                              sigma=sigma, option=option, direction=direction, value=value)
        
        self._return_options(legs=4, S0=self.S0, K1=self.K1, K2=self.K2, K3=self.K3, 
                             K4=self.K4, T1=self.T, T2=self.T, T3=self.T, T4=self.T, 
                             r=self.r, q=self.q, sigma=self.sigma, option1=self.option, 
                             option2=self.option, option3=self.option, option4=self.option)
        
        if self.direction == 'long':
            payoff = (self.C1 - self.C2 - self.C3 + self.C4 - self.C1_0 + 
                      self.C2_0 + self.C3_0 - self.C4_0)
            if self.value == True:
                payoff2 = (self.C1_G - self.C2_G - self.C3_G + self.C4_G - self.C1_0 + 
                           self.C2_0 + self.C3_0 - self.C4_0)
            else:
                payoff2 = None
        
        if self.direction == 'short':
            payoff = (-self.C1 + self.C2 + self.C3 - self.C4 + self.C1_0 - 
                      self.C2_0 - self.C3_0 + self.C4_0)
            if self.value == True:
                payoff2 = (-self.C1_G + self.C2_G + self.C3_G - self.C4_G + self.C1_0 - 
                           self.C2_0 - self.C3_0 + self.C4_0)
            else:
                payoff2 = None
                
        if self.option == 'call' and self.direction == 'long':
            title = 'Long Iron Condor with Calls'
        if self.option == 'put' and self.direction == 'long':
            title = 'Long Iron Condor with Puts'
        if self.option == 'call' and self.direction == 'short':
            title = 'Short Iron Condor with Calls'
        if self.option == 'put' and self.direction == 'short':
            title = 'Short Iron Condor with Puts'    
       
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
    
    
    
    def _return_options(self, legs=2, S0=None, K1=None, K2=None, K3=None, 
                        K4=None, T1=None, T2=None, T3=None, T4=None, r=None, q=None, 
                        sigma=None, option1=None, option2=None, option3=None, option4=None):
 
        self.SA = np.linspace(0.8 * self.S0, 1.2 * self.S0, 100)
        
        self.C1_0 = self.price(S=S0, K=K1, T=T1, r=r, q=q, sigma=sigma, option=option1, 
                               combo=True)
        self.C1 = self.price(S=self.SA, K=K1, T=0, r=r, q=q, sigma=sigma, option=option1, 
                             combo=True)
        self.C1_G = self.price(S=self.SA, K=K1, T=T1, r=r, q=q, sigma=sigma, option=option1, 
                               combo=True)
        
        if legs > 1:
            self.C2_0 = self.price(S=S0, K=K2, T=T2, r=r, q=q, sigma=sigma, option=option2, 
                                   combo=True)
            self.C2 = self.price(S=self.SA, K=K2, T=0, r=r, q=q, sigma=sigma, option=option2, 
                                 combo=True)
            self.C2_G = self.price(S=self.SA, K=K2, T=T2, r=r, q=q, sigma=sigma, 
                                   option=option2, combo=True)

        if legs > 2:
            self.C3_0 = self.price(S=self.S0, K=K3, T=T3, r=r, q=q, sigma=sigma, 
                                   option=option3, combo=True)
            self.C3 = self.price(S=self.SA, K=K3, T=0, r=r, q=q, sigma=sigma, option=option3, 
                                 combo=True)
            self.C3_G = self.price(S=self.SA, K=K3, T=T3, r=r, q=q, sigma=sigma, 
                                   option=option3, combo=True)
        
        if legs > 3:
            self.C4_0 = self.price(S=self.S0, K=K4, T=T4, r=r, q=q, sigma=sigma, 
                                   option=option4, combo=True)
            self.C4 = self.price(S=self.SA, K=K4, T=0, r=r, q=q, sigma=sigma, option=option4, 
                                 combo=True)
            self.C4_G = self.price(S=self.SA, K=K4, T=T4, r=r, q=q, sigma=sigma, 
                                   option=option4, combo=True)
        
        return self
        
    
