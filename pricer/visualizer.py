import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot

# Dictionary of default parameters
df_dict = {'df_S':100, 
           'df_S0':100,
           'df_K':100,
           'df_K1':95,
           'df_K2':105,
           'df_K3':105,
           'df_K4':105,
           'df_G1':90,
           'df_G2':100,
           'df_G3':110,
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
           'df_refresh':'Std',
           'df_combo_payoff':'straddle',
           'df_delta_shift':25,
           'df_delta_shift_type':'avg',
           'df_greek':'delta',
           'df_interactive':False,
           'df_notebook':True,
           'df_colorscheme':'jet',
           'df_colorintensity':1,
           'df_size':(12, 10),
           'df_graphtype':'2D',
           'df_y_plot':'delta',
           'df_x_plot':'time',
           'df_time_shift':0.25,
           'df_cash':False,
           'df_axis':'price',

            # List of default parameters used when refreshing 
            'df_params_list':['S', 'S0', 'K', 'K1', 'K2', 'K3', 'K4', 'G1', 
                              'G2', 'G3', 'H', 'R', 'T', 'T1', 'T2', 'T3', 'T4', 
                              'r', 'b', 'q', 'sigma', 'eta', 'phi', 'barrier_direction', 
                              'knock', 'option', 'option1', 'option2', 'option3', 
                              'option4', 'direction', 'value', 'ratio', 'refresh', 
                              'delta_shift', 'delta_shift_type', 'greek', 'interactive', 
                              'notebook', 'colorscheme', 'colorintensity', 'size', 
                              'graphtype', 'cash', 'axis'],
            
            # List of Greeks where call and put values are the same
            'df_equal_greeks':['gamma', 'vega', 'vomma', 'vanna', 'zomma', 'speed', 
                              'color', 'ultima', 'vega bleed'],
            
            # Payoffs requiring changes to default parameters
            'df_mod_payoffs':['collar', 'straddle', 'butterfly', 'christmas tree',
                              'condor', 'iron butterfly', 'iron condor'],
            
            # Those parameters that need changing
            'df_mod_params':['S0', 'K', 'K1', 'K2', 'K3', 'K4'],
            
            # Combo parameter values differing from standard defaults
            'df_combo_dict':{'collar':{'S0':100,
                                       'K':100,
                                       'K1':98,
                                       'K2':102},
                             'straddle':{'S0':100,
                                         'K':100,
                                         'K1':100,
                                         'K2':100},
                             'butterfly':{'S0':100,
                                          'K':100,
                                          'K1':95,
                                          'K2':100,
                                          'K3':105,
                                          'K4':105},
                             'christmas tree':{'S0':100,
                                               'K':100,
                                               'K1':95,
                                               'K2':100,
                                               'K3':105,
                                               'K4':105},
                             'condor':{'S0':100,
                                       'K':100,
                                       'K1':90,
                                       'K2':95,
                                       'K3':100,
                                       'K4':105},
                             'iron butterfly':{'S0':100,
                                               'K':100,
                                               'K1':95,
                                               'K2':100,
                                               'K3':100,
                                               'K4':105},
                             'iron condor':{'S0':100,
                                            'K':100,
                                            'K1':90,
                                            'K2':95,
                                            'K3':100,
                                            'K4':105}},
            
            # Dictionary mapping function parameters to x axis labels for 2D graphs
            'df_x_name_dict':{'price':'SA', 
                              'strike':'SA',
                              'vol':'sigmaA', 
                              'time':'TA'},
            
            # Dictionary mapping scaling parameters to x axis labels for 2D graphs
            'df_x_scale_dict':{'price':1, 
                               'strike':1,
                               'vol':100, 
                               'time':365},
            
            # Dictionary mapping function parameters to y axis labels for 2D graphs
            'df_y_name_dict':{'value':'price', 
                              'delta':'delta', 
                              'gamma':'gamma', 
                              'vega':'vega', 
                              'theta':'theta'},

            # Dictionary mapping function parameters to axis labels for 3D graphs
            'df_label_dict':{'price':'Underlying Price',
                             'value':'Theoretical Value',
                             'vol':'Volatility %',
                             'time':'Time to Expiration (Days)',
                             'delta':'Delta',
                             'gamma':'Gamma',
                             'vega':'Vega',
                             'theta':'Theta',
                             'rho':'Rho',
                             'strike':'Strike Price'},
            
            # Ranges of Underlying price and Time to Expiry for 3D greeks graphs
            'df_3D_chart_ranges':{'price':{'SA_lower':0.8,
                                           'SA_upper':1.2,
                                           'TA_lower':0.01,
                                           'TA_upper':1},
                                  'delta':{'SA_lower':0.25,
                                           'SA_upper':1.75,
                                           'TA_lower':0.01,
                                           'TA_upper':2},
                                  'gamma':{'SA_lower':0.8,
                                           'SA_upper':1.2,
                                           'TA_lower':0.01,
                                           'TA_upper':5},
                                  'vega':{'SA_lower':0.5,
                                          'SA_upper':1.5,
                                          'TA_lower':0.01,
                                          'TA_upper':1},
                                  'theta':{'SA_lower':0.8,
                                           'SA_upper':1.2,
                                           'TA_lower':0.01,
                                           'TA_upper':1},
                                  'rho':{'SA_lower':0.8,
                                         'SA_upper':1.2,
                                         'TA_lower':0.01,
                                         'TA_upper':0.5},
                                  'vomma':{'SA_lower':0.5,
                                           'SA_upper':1.5,
                                           'TA_lower':0.01,
                                           'TA_upper':1},
                                  'vanna':{'SA_lower':0.5,
                                           'SA_upper':1.5,
                                           'TA_lower':0.01,
                                           'TA_upper':1},
                                  'zomma':{'SA_lower':0.8,
                                           'SA_upper':1.2,
                                           'TA_lower':0.01,
                                           'TA_upper':0.5},
                                  'speed':{'SA_lower':0.8,
                                           'SA_upper':1.2,
                                           'TA_lower':0.01,
                                           'TA_upper':0.5},
                                  'color':{'SA_lower':0.8,
                                           'SA_upper':1.2,
                                           'TA_lower':0.01,
                                           'TA_upper':0.5},
                                  'ultima':{'SA_lower':0.5,
                                            'SA_upper':1.5,
                                            'TA_lower':0.01,
                                            'TA_upper':1},
                                  'vega bleed':{'SA_lower':0.5,
                                                'SA_upper':1.5,
                                                'TA_lower':0.01,
                                                'TA_upper':1},
                                  'charm':{'SA_lower':0.8,
                                           'SA_upper':1.2,
                                           'TA_lower':0.01,
                                           'TA_upper':0.25}},
            
            # Greek names as function input and individual function names            
            'df_greek_dict':{'price':'price',
                             'delta':'delta',
                             'gamma':'gamma',
                             'vega':'vega',
                             'theta':'theta',
                             'rho':'rho',
                             'vomma':'vomma',
                             'vanna':'vanna',
                             'zomma':'zomma',
                             'speed':'speed',
                             'color':'color',
                             'ultima':'ultima',
                             'vega bleed':'vega_bleed',
                             'charm':'charm'}}



class Option():
    
    def __init__(self, S=df_dict['df_S'], S0=df_dict['df_S0'], K=df_dict['df_K'], 
                 K1=df_dict['df_K1'], K2=df_dict['df_K2'], K3=df_dict['df_K3'], 
                 K4=df_dict['df_K4'], G1=df_dict['df_G1'], G2=df_dict['df_G2'], 
                 G3=df_dict['df_G3'], H=df_dict['df_H'], R=df_dict['df_R'], T=df_dict['df_T'], 
                 T1=df_dict['df_T1'], T2=df_dict['df_T2'], T3=df_dict['df_T3'], 
                 T4=df_dict['df_T4'],r=df_dict['df_r'], b=df_dict['df_b'], 
                 q=df_dict['df_q'], sigma=df_dict['df_sigma'], eta=df_dict['df_eta'], 
                 phi=df_dict['df_phi'], barrier_direction=df_dict['df_barrier_direction'], 
                 knock=df_dict['df_knock'], option=df_dict['df_option'], option1=df_dict['df_option1'], 
                 option2=df_dict['df_option2'], option3=df_dict['df_option3'], 
                 option4=df_dict['df_option4'], direction=df_dict['df_direction'], 
                 value=df_dict['df_value'], ratio=df_dict['df_ratio'], refresh=df_dict['df_refresh'], 
                 combo_payoff=df_dict['df_combo_payoff'], delta_shift=df_dict['df_delta_shift'], 
                 delta_shift_type=df_dict['df_delta_shift_type'], greek=df_dict['df_greek'], 
                 interactive=df_dict['df_interactive'], notebook=df_dict['df_notebook'], 
                 colorscheme=df_dict['df_colorscheme'], colorintensity=df_dict['df_colorintensity'], 
                 size=df_dict['df_size'], graphtype=df_dict['df_graphtype'], y_plot=df_dict['df_y_plot'], 
                 x_plot=df_dict['df_x_plot'], x_name_dict=df_dict['df_x_name_dict'], 
                 x_scale_dict=df_dict['df_x_scale_dict'], y_name_dict=df_dict['df_y_name_dict'], 
                 time_shift=df_dict['df_time_shift'], cash=df_dict['df_cash'], axis=df_dict['df_axis'], 
                 df_combo_dict=df_dict['df_combo_dict'], df_params_list=df_dict['df_params_list'], 
                 equal_greeks=df_dict['df_equal_greeks'], mod_payoffs=df_dict['df_mod_payoffs'], 
                 mod_params=df_dict['df_mod_params'], label_dict=df_dict['df_label_dict'], 
                 greek_dict=df_dict['df_greek_dict'], df_dict=df_dict):

        self.S = S # Spot price
        self.S0 = S0 # Spot price
        self.K = K # Strike price
        self.K1 = K1 # Strike price for combo payoffs
        self.K2 = K2 # Strike price for combo payoffs
        self.K3 = K3 # Strike price for combo payoffs
        self.K4 = K4 # Strike price for combo payoffs
        self.G1 = G1 # Strike price for 2D Greeks graphs
        self.G2 = G2 # Strike price for 2D Greeks graphs
        self.G3 = G3 # Strike price for 2D Greeks graphs
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
        self.refresh = refresh # Flag whether to refresh default values in price formula
        self.delta_shift = delta_shift # Size of shift used in shift_delta function
        self.delta_shift_type = delta_shift_type # Shift type - Up, Down or Avg
        self.df_dict = df_dict # Dictionary of parameter defaults
        self.df_combo_dict = df_combo_dict # Dictionary of payoffs with different default parameters
        self.df_params_list = df_params_list # List of default parameters
        self.greek = greek # Option greek to display e.g. delta
        self.interactive = interactive # Whether to display static mpl 3D graph or plotly interactive graph
        self.notebook = notebook # Whether running in iPython notebook or not, False creates a popup html page 
        self.colorscheme = colorscheme # Color palette to use in 3D graphs
        self.colorintensity = colorintensity # Alpha level to use in 3D graphs
        self.size = size # Tuple for size of 3D static graph
        self.graphtype = graphtype # 2D or 3D graph 
        self.y_plot = y_plot # X-axis in 2D greeks graph
        self.x_plot = x_plot # Y-axis in 2D greeks graph
        self.x_name_dict = x_name_dict # Dictionary mapping function parameters to x axis labels for 2D graphs
        self.x_scale_dict = x_scale_dict # Dictionary mapping scaling parameters to x axis labels for 2D graphs
        self.y_name_dict = y_name_dict # Dictionary mapping function parameters to y axis labels for 2D graphs
        self.time_shift = time_shift # Time between periods used in 2D greeks graph
        self.cash = cash # Whether to graph forward at cash or discount
        self.axis = axis # Price or Vol against Time in 3D graphs
        self.mod_payoffs = mod_payoffs # Combo payoffs needing different default parameters
        self.mod_params = mod_params # Parameters of these payoffs that need changing
        self.label_dict = label_dict # Dictionary mapping function parameters to axis labels
        self.equal_greeks = equal_greeks # List of Greeks where call and put values are the same
        self.greek_dict = greek_dict # Greek names as function input and individual function names
        self.combo_payoff = combo_payoff # 2D graph payoff structure

    
    def _initialise_func(self, **kwargs):
        """
        

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self._refresh_params(**kwargs)
        self._refresh_dist()
        
        return self


    def _initialise_graphs(self, **kwargs):
        """
        

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self._set_params(**kwargs)
        self._refresh_dist()
        
        return self
    

    def _initialise_barriers(self, **kwargs):
        """
        

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self._refresh_params(**kwargs)
        self._refresh_dist()
        self._barrier_factors()

        return self


    def _refresh_params(self, **kwargs):
        """
        

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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
        

        for key in list(set(self.df_params_list) - set(kwargs.keys())):
            if key not in kwargs:
                val = df_dict['df_'+str(key)]
                self.__dict__[key] = val
                
        return self        
   
    
    def _set_params(self, **kwargs):
        """
        

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v
    
        return self
       
        
    def _refresh_dist(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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
              refresh=None):
        """
        Return the Black Scholes Option Price

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 

        Returns
        -------
        Float
            Black Scholes Option Price. If combo is set to true the price to be used 
            in combo graphs so the distributions are refreshed but not the parameters.

        """
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
        
        if self.option == "call":
            self.opt_price = ((self.S * self.carry * self.Nd1) - 
                              (self.K * np.exp(-self.r * self.T) * self.Nd2))  
        if self.option == 'put':
            self.opt_price = ((self.K * np.exp(-self.r * self.T) * self.minusNd2) - 
                              (self.S * self.carry * self.minusNd1))
        
        np.nan_to_num(self.opt_price, copy=False)
                
        return self.opt_price


    def delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        Sensitivity of the option price to changes in asset price

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 


        Returns
        -------
        TYPE
            DESCRIPTION.

        """      
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
                                
        if self.option == 'call':
            self.opt_delta = self.carry * self.Nd1
        if self.option == 'put':
            self.opt_delta = self.carry * (self.Nd1 - 1)
            
        return self.opt_delta
    
    
    def shift_delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
                    shift=None, shift_type=None, refresh=None):
        """
        Sensitivity of the option price to changes in asset price
        Calculated by taking the difference in price for varying shift sizes

        Parameters
        ----------
        S : TYPE, optional
            DESCRIPTION. The default is None.
        K : TYPE, optional
            DESCRIPTION. The default is None.
        T : TYPE, optional
            DESCRIPTION. The default is None.
        r : TYPE, optional
            DESCRIPTION. The default is None.
        q : TYPE, optional
            DESCRIPTION. The default is None.
        sigma : TYPE, optional
            DESCRIPTION. The default is None.
        option : TYPE, optional
            DESCRIPTION. The default is None.
        shift : TYPE, optional
            DESCRIPTION. The default is None.
        shift_type : TYPE, optional
            DESCRIPTION. The default is None.
        refresh : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                  shift=shift, shift_type=shift_type)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option,
                                    shift=shift, shift_type=shift_type, refresh=refresh)
        
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
    
    
    def theta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        Sensitivity of the option price to changes in time to maturity

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 


        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
                   
        if self.option == 'call':
            self.opt_theta = ((-self.S * self.carry * self.nd1 * self.sigma ) / 
                              (2 * np.sqrt(self.T)) - (self.b - self.r) * self.S * self.carry * 
                              self.Nd1 - self.r * self.K * np.exp(-self.r * self.T) * self.Nd2)
        if self.option == 'put':   
            self.opt_theta = ((-self.S * self.carry * self.nd1 * self.sigma ) / 
                              (2 * np.sqrt(self.T)) + (self.b - self.r) * self.S * self.carry * 
                              self.minusNd1 + self.r * self.K * np.exp(-self.r * self.T) * self.minusNd2)

        return self.opt_theta
    
    
    def gamma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        How much delta will change due to a small change in the underlying asset price

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 


        Returns
        -------
        TYPE
            DESCRIPTION.

        """
               
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_gamma = ((self.nd1 * self.carry) / (self.S * self.sigma * np.sqrt(self.T)))
        
        return self.opt_gamma
    
    
    def vega(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
             refresh=None):
        """
        Sensitivity of the option price to changes in volatility

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)

        self.opt_vega = self.S * self.carry * self.nd1 * np.sqrt(self.T)
        
        return self.opt_vega
    
    
    def rho(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
            refresh=None):
        """
        Sensitivity of the option price to changes in the risk free rate

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
        
        if self.option == 'call':
            self.opt_rho = self.T * self.K * np.exp(-self.r * self.T) * self.Nd2
        if self.option == 'put':
            self.opt_rho = -self.T * self.K * np.exp(-self.r * self.T) * self.minusNd2
            
        return self.opt_rho


    def vanna(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, refresh=None):
        """
        DdeltaDvol, DvegaDspot 
        How much delta will change due to a small change in volatility
        How much vega will change due to a small change in the asset price   

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
              
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_vanna = ((-self.carry * self.d2) / self.sigma) * self.nd1 

        return self.opt_vanna               
           

    def charm(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        DdeltaDtime, Delta Bleed 
        How much delta will change due to a small change in time to expiration

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
               
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
        
        if self.option == 'call':
            self.opt_charm = (-self.carry * ((self.nd1 * ((self.b / (self.sigma * np.sqrt(self.T))) - 
                                                          (self.d2 / (2 * self.T)))) + 
                                             ((self.b - self.r) * self.Nd1)))
        if self.option == 'put':
            self.opt_charm = (-self.carry * ((self.nd1 * ((self.b / (self.sigma * np.sqrt(self.T))) - 
                                                          (self.d2 / (2 * self.T)))) - 
                                             ((self.b - self.r) * self.minusNd1)))
        return self.opt_charm
               

    def zomma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        DgammaDvol
        Sensitivity of gamma to changes in volatility
        
        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_zomma = (self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                     self.option, self.refresh) * ((self.d1 * self.d2 - 1) / self.sigma))
        
        return self.opt_zomma


    def speed(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        DgammaDspot
        Sensitivity of gamma to changes in asset price 
        3rd derivative of option price with respect to spot

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_speed = -(self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                      self.option, self.refresh) * (1 + (self.d1 / (self.sigma * 
                                      np.sqrt(self.T)))) / self.S)
        
        return self.opt_speed


    def color(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        DgammaDtime, Gamma Bleed, Gamma Theta
        Sensitivity of gamma to changes in time to maturity

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_color = (self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                     self.option, self.refresh) * ((self.r - self.b) + 
                                    ((self.b * self.d1) / (self.sigma * np.sqrt(self.T))) + 
                                    ((1 - self.d1 * self.d2) / (2 * self.T))))
        
        return self.opt_color


    def vomma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        DvegaDvol, Vega Convexity, Volga, Vol Gamma
        How much vega will change due to a small change in implied volatility

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
       
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_vomma = (self.vega(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                    self.option, self.refresh) * ((self.d1 * self.d2) / (self.sigma)))
        
        return self.opt_vomma


    def ultima(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
               refresh=None):
        """
        DvommaDvol
        How much vomma will change due to a small change in implied volatility
        3rd derivative of option price wrt volatility

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_ultima = (self.vomma(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                      self.option, self.refresh) * ((1 / self.sigma) * 
                                     (self.d1 * self.d2 - (self.d1 / self.d2) - 
                                     (self.d2 / self.d1) - 1)))
        
        return self.opt_ultima


    def vega_bleed(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
                   refresh=None):
        """
        DvegaDtime
        How much vega will change due to a small change in time to expiration

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_vega_bleed = (self.vega(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                               self.option, self.refresh) * (self.r - self.b + 
                             ((self.b * self.d1) / (self.sigma * np.sqrt(self.T))) - 
                              ((1 + (self.d1 * self.d2) ) / (2 * self.T))))

        return self.opt_vega_bleed



    def barrier_price(self, S=None, K=None, H=None, R=None, T=None, r=None, q=None, 
                       sigma=None, barrier_direction=None, knock=None, option=None, 
                       refresh=None):
        """
  
    
        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        H : TYPE, optional
            DESCRIPTION. The default is None.
        R : TYPE, optional
            DESCRIPTION. The default is None.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        barrier_direction : TYPE, optional
            DESCRIPTION. The default is None.
        knock : TYPE, optional
            DESCRIPTION. The default is None.
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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


    def _strike_tenor_label(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.strike_label = dict()
        for key, value in {'G1':'label1', 'G2':'label2', 'G3':'label3'}.items():
            if self.__dict__[str(key)] == self.S0:
                self.strike_label[value] = 'ATM Strike'
            else:
                self.strike_label[value] = str(int(self.__dict__[key]))+' Strike' 
               
        for k, v in {'T1':'label1', 'T2':'label2', 'T3':'label3'}.items():
            self.__dict__[v] = str(int(self.__dict__[str(k)]*365))+' Day '+self.strike_label[str(v)]
                
        return self            


    def _graph_space_prep(self, axis='price'):
        """
        

        Parameters
        ----------
        axis : TYPE, optional
            DESCRIPTION. The default is 'price'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.SA_lower = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['SA_lower']
        self.SA_upper = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['SA_upper']
        self.TA_lower = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['TA_lower']
        self.TA_upper = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['TA_upper']
        self.sigmaA_lower = 0.05 
        self.sigmaA_upper = 0.5 

        self.SA = np.linspace(self.SA_lower * self.S0, self.SA_upper * self.S0, 100)
        self.TA = np.linspace(self.TA_lower, self.TA_upper, 100)
        self.sigmaA = np.linspace(self.sigmaA_lower, self.sigmaA_upper, 100)

        self.ymin = self.TA_lower
        self.ymax = self.TA_upper
        self.axis_label2 = 'Time to Expiration (Days)'
        
        if axis == 'price':
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.xmin = self.SA_lower
            self.xmax = self.SA_upper
            self.graph_scale = 1
            self.axis_label1 = 'Underlying Value'            
            
        if axis == 'vol':
            self.x, self.y = np.meshgrid(self.sigmaA, self.TA)
            self.xmin = self.sigmaA_lower
            self.xmax = self.sigmaA_upper    
            self.graph_scale = 100
            self.axis_label1 = 'Volatility %'    

        return self


    def _vis_payoff(self, S0=None, SA=None, payoff=None, label=None, title='Option Payoff', 
                    payoff2=None, label2=None, payoff3=None, label3=None, payoff4=None, 
                    label4=None, xlabel='Stock Price', ylabel='P&L'):
        """
        

        Parameters
        ----------
        S0 : TYPE, optional
            DESCRIPTION. The default is None.
        SA : TYPE, optional
            DESCRIPTION. The default is None.
        payoff : TYPE, optional
            DESCRIPTION. The default is None.
        label : TYPE, optional
            DESCRIPTION. The default is None.
        title : TYPE, optional
            DESCRIPTION. The default is 'Option Payoff'.
        payoff2 : TYPE, optional
            DESCRIPTION. The default is None.
        label2 : TYPE, optional
            DESCRIPTION. The default is None.
        payoff3 : TYPE, optional
            DESCRIPTION. The default is None.
        label3 : TYPE, optional
            DESCRIPTION. The default is None.
        payoff4 : TYPE, optional
            DESCRIPTION. The default is None.
        label4 : TYPE, optional
            DESCRIPTION. The default is None.
        xlabel : TYPE, optional
            DESCRIPTION. The default is 'Stock Price'.
        ylabel : TYPE, optional
            DESCRIPTION. The default is 'P&L'.

        Returns
        -------
        None.

        """
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
    
    
    def _vis_greeks_mpl(self, xarray1=None, xarray2=None, xarray3=None, 
                        xarray4=None, yarray1=None, yarray2=None, yarray3=None, 
                        yarray4=None, label1=None, label2=None, label3=None, label4=None, 
                        xlabel=None, ylabel=None, title='Payoff'):
        """
        

        Parameters
        ----------
        xarray1 : TYPE, optional
            DESCRIPTION. The default is None.
        xarray2 : TYPE, optional
            DESCRIPTION. The default is None.
        xarray3 : TYPE, optional
            DESCRIPTION. The default is None.
        xarray4 : TYPE, optional
            DESCRIPTION. The default is None.
        yarray1 : TYPE, optional
            DESCRIPTION. The default is None.
        yarray2 : TYPE, optional
            DESCRIPTION. The default is None.
        yarray3 : TYPE, optional
            DESCRIPTION. The default is None.
        yarray4 : TYPE, optional
            DESCRIPTION. The default is None.
        label1 : TYPE, optional
            DESCRIPTION. The default is None.
        label2 : TYPE, optional
            DESCRIPTION. The default is None.
        label3 : TYPE, optional
            DESCRIPTION. The default is None.
        label4 : TYPE, optional
            DESCRIPTION. The default is None.
        xlabel : TYPE, optional
            DESCRIPTION. The default is None.
        ylabel : TYPE, optional
            DESCRIPTION. The default is None.
        title : TYPE, optional
            DESCRIPTION. The default is 'Payoff'.

        Returns
        -------
        None.

        """
        fig, ax = plt.subplots()
        plt.style.use('seaborn-darkgrid')
        ax.plot(xarray1, yarray1, color='blue', label=label1)
        ax.plot(xarray2, yarray2, color='red', label=label2)
        ax.plot(xarray3, yarray3, color='green', label=label3)
        if label4 is not None:
            ax.plot(xarray4, yarray4, color='orange', label=label4)
        plt.grid(True)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend()
        plt.show()
   
    
    def greeks(self, greek=None, x_plot=None, y_plot=None, S0=None, G1=None, G2=None, 
               G3=None, T=None, T1=None, T2=None, T3=None, time_shift=None, r=None, 
               q=None, sigma=None, option=None, direction=None, interactive=None, 
               notebook=None, colorscheme=None, colorintensity=None, size=None, 
               axis=None, graphtype=None):
        """
        

        Parameters
        ----------
        greek : TYPE, optional
            DESCRIPTION. The default is None.
        x_plot : TYPE, optional
            DESCRIPTION. The default is None.
        y_plot : TYPE, optional
            DESCRIPTION. The default is None.
        S0 : TYPE, optional
            DESCRIPTION. The default is None.
        G1 : TYPE, optional
            DESCRIPTION. The default is None.
        G2 : TYPE, optional
            DESCRIPTION. The default is None.
        G3 : TYPE, optional
            DESCRIPTION. The default is None.
        T : TYPE, optional
            DESCRIPTION. The default is None.
        T1 : TYPE, optional
            DESCRIPTION. The default is None.
        T2 : TYPE, optional
            DESCRIPTION. The default is None.
        T3 : TYPE, optional
            DESCRIPTION. The default is None.
        time_shift : TYPE, optional
            DESCRIPTION. The default is None.
        r : TYPE, optional
            DESCRIPTION. The default is None.
        q : TYPE, optional
            DESCRIPTION. The default is None.
        sigma : TYPE, optional
            DESCRIPTION. The default is None.
        option : TYPE, optional
            DESCRIPTION. The default is None.
        direction : TYPE, optional
            DESCRIPTION. The default is None.
        interactive : TYPE, optional
            DESCRIPTION. The default is None.
        notebook : TYPE, optional
            DESCRIPTION. The default is None.
        colorscheme : TYPE, optional
            DESCRIPTION. The default is None.
        colorintensity : TYPE, optional
            DESCRIPTION. The default is None.
        size : TYPE, optional
            DESCRIPTION. The default is None.
        axis : TYPE, optional
            DESCRIPTION. The default is None.
        graphtype : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._initialise_func(greek=greek, x_plot=x_plot, y_plot=y_plot, S0=S0, G1=G1, 
                              G2=G2, G3=G3, T=T, T1=T1, T2=T2, T3=T3, time_shift=time_shift,
                              r=r, q=q, sigma=sigma, option=option, interactive=interactive, 
                              notebook=notebook, colorscheme=colorscheme, colorintensity=colorintensity, 
                              size=size, direction=direction, axis=axis, graphtype=graphtype)
        
        
        if self.graphtype == '2D':
            self.greeks_graphs_2D(x_plot=self.x_plot, y_plot=self.y_plot, 
                                  S0=self.S0, G1=self.G1, G2=self.G2, G3=self.G3, 
                                  T=self.T, T1=self.T1, T2=self.T2, T3=self.T3, 
                                  time_shift=self.time_shift, r=self.r, q=self.q, 
                                  sigma=self.sigma, option=self.option, direction=self.direction)
            
        if self.graphtype == '3D':
            self.greeks_graphs_3D(greek=self.greek, S0=self.S0, r=self.r, q=self.q, sigma=self.sigma, 
                                  option=self.option, interactive=self.interactive, notebook=self.notebook, 
                                  colorscheme=self.colorscheme, colorintensity=self.colorintensity, 
                                  size=self.size, direction=self.direction, axis=self.axis)
    
    
    def greeks_graphs_3D(self, greek=None, S0=None, r=None, q=None, sigma=None, 
                         option=None, interactive=None, notebook=None, colorscheme=None, 
                         colorintensity=None, size=None, direction=None, axis=None):
        """
        

        Parameters
        ----------
        greek : TYPE, optional
            DESCRIPTION. The default is None.
        S0 : TYPE, optional
            DESCRIPTION. The default is None.
        r : TYPE, optional
            DESCRIPTION. The default is None.
        q : TYPE, optional
            DESCRIPTION. The default is None.
        sigma : TYPE, optional
            DESCRIPTION. The default is None.
        option : TYPE, optional
            DESCRIPTION. The default is None.
        interactive : TYPE, optional
            DESCRIPTION. The default is None.
        notebook : TYPE, optional
            DESCRIPTION. The default is None.
        colorscheme : TYPE, optional
            DESCRIPTION. The default is None.
        colorintensity : TYPE, optional
            DESCRIPTION. The default is None.
        size : TYPE, optional
            DESCRIPTION. The default is None.
        direction : TYPE, optional
            DESCRIPTION. The default is None.
        axis : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._initialise_func(greek=greek, S0=S0, r=r, q=q, sigma=sigma, option=option, 
                              interactive=interactive, notebook=notebook, colorscheme=colorscheme, 
                              colorintensity=colorintensity, size=size, direction=direction, 
                              axis=axis)
        

        for greek_label, greek_func in self.greek_dict.items():
            if self.greek in self.equal_greeks:
                self.option = 'Call / Put'
            if self.greek == greek_label:
                if self.axis == 'price':
                    self._graph_space_prep(axis='price')
                    self.z = getattr(self, greek_func)(S=self.x, K=self.S0, T=self.y, 
                                                       r=self.r, sigma=self.sigma, 
                                                       option=self.option, refresh='graph')
                if self.axis == 'vol':
                    self._graph_space_prep(axis='vol')
                    self.z = getattr(self, greek_func)(S=self.S0, K=self.S0, T=self.y, 
                                                       r=self.r, sigma=self.x, 
                                                       option=self.option, refresh='graph')
                    
        self._vis_greeks_3D()            
    
   
    def _vis_greeks_3D(self):
        """
        

        Returns
        -------
        None.

        """
        if self.direction == 'short':
            self.z = -self.z
        
        
        if self.option == 'Call / Put':
            titlename = str(str(self.direction.title())+' '+self.option+' Option '+str(self.greek.title()))
        else:    
            titlename = str(str(self.direction.title())+' '+str(self.option.title())+
                            ' Option '+str(self.greek.title()))
           

        if self.interactive == False:
        
            fig = plt.figure(figsize=self.size)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.x * self.graph_scale,
                            self.y * 365,
                            self.z,
                            rstride=2, cstride=2,
                            cmap=plt.get_cmap(self.colorscheme),
                            alpha=self.colorintensity,
                            linewidth=0.25)
            ax.set_zlim(auto=True)
            ax.invert_xaxis()
            ax.set_xlabel(self.axis_label1, fontsize=12)
            ax.set_ylabel(self.axis_label2, fontsize=12)
            ax.set_zlabel(str(self.greek.title()), fontsize=12)
            ax.set_title(titlename, fontsize=14)
            plt.show()


        if self.interactive == True:
            
            contour_x_start = self.ymin
            contour_x_stop = self.ymax * 360
            contour_x_size = contour_x_stop / 18
            contour_y_start = self.xmin
            contour_y_stop = self.xmax * self.graph_scale
            contour_y_size = int((self.xmax - self.xmin) / 20)
            contour_z_start = np.min(self.z)
            contour_z_stop = np.max(self.z)
            contour_z_size = int((np.max(self.z) - np.min(self.z)) / 10)
            
            
            fig = go.Figure(data=[go.Surface(x=self.y*365, 
                                             y=self.x*self.graph_scale, 
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
                                xaxis_title=self.axis_label2,
                                yaxis_title=self.axis_label1,
                                zaxis_title=str(self.greek.title()),),
                              title=titlename, autosize=False, 
                              width=800, height=800,
                              margin=dict(l=65, r=50, b=65, t=90),
                             scene_camera=camera)
            
            if self.notebook == True:
                fig.show()
            else:
                plot(fig, auto_open=True)
 
    
    
    def greeks_graphs_2D(self, y_plot=None, x_plot=None, S0=None, G1=None, G2=None, 
                             G3=None, T=None, T1=None, T2=None, T3=None, time_shift=None, 
                             r=None, q=None, sigma=None, option=None, direction=None):
        """
        

        Parameters
        ----------
        x_plot : TYPE, optional
            DESCRIPTION. The default is None.
        y_plot : TYPE, optional
            DESCRIPTION. The default is None.
        S0 : TYPE, optional
            DESCRIPTION. The default is None.
        G1 : TYPE, optional
            DESCRIPTION. The default is None.
        G2 : TYPE, optional
            DESCRIPTION. The default is None.
        G3 : TYPE, optional
            DESCRIPTION. The default is None.
        T : TYPE, optional
            DESCRIPTION. The default is None.
        T1 : TYPE, optional
            DESCRIPTION. The default is None.
        T2 : TYPE, optional
            DESCRIPTION. The default is None.
        T3 : TYPE, optional
            DESCRIPTION. The default is None.
        time_shift : TYPE, optional
            DESCRIPTION. The default is None.
        r : TYPE, optional
            DESCRIPTION. The default is None.
        q : TYPE, optional
            DESCRIPTION. The default is None.
        sigma : TYPE, optional
            DESCRIPTION. The default is None.
        option : TYPE, optional
            DESCRIPTION. The default is None.
        direction : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._initialise_func(x_plot=x_plot, y_plot=y_plot, S0=S0, G1=G1, G2=G2, 
                              G3=G3, T=T, T1=T1, T2=T2, T3=T3, time_shift=time_shift, 
                              r=r, q=q, sigma=sigma, option=option, direction=direction)
        
        self.SA = np.linspace(0.8 * self.S0, 1.2 * self.S0, 100)
        self.sigmaA = np.linspace(0.05, 0.5, 100)
        self.TA = np.linspace(0.01, 1, 100)
    
        self._2D_general_graph()       
    
          
    
    

    def _2D_general_graph(self):                               
        """
        

        Returns
        -------
        None.

        """
        if self.y_plot in self.y_name_dict.keys():
            for opt in [1, 2, 3]:
                if self.x_plot == 'price':
                    self.__dict__['C'+str(opt)] = getattr(self, self.y_name_dict[self.y_plot])(S=self.SA, K=self.__dict__['G'+str(opt)], T=self.__dict__['T'+str(opt)], 
                                                                             r=self.r, q=self.q, sigma=self.sigma, option=self.option, refresh='graph')
                if self.x_plot == 'vol':
                    self.__dict__['C'+str(opt)] = getattr(self, self.y_name_dict[self.y_plot])(S=self.S0, K=self.__dict__['G'+str(opt)], T=self.__dict__['T'+str(opt)], 
                                                                              r=self.r, q=self.q, sigma=self.sigmaA, option=self.option, refresh='graph')
                if self.x_plot == 'time':        
                    self.__dict__['C'+str(opt)] = getattr(self, self.y_name_dict[self.y_plot])(S=self.S0, K=self.__dict__['G'+str(opt)], T=self.TA, r=self.r, 
                                                                                   q=self.q, sigma=self.sigma, option=self.option, refresh='graph')
        
            if self.direction == 'short':
                for opt in [1, 2, 3]:
                    self.__dict__['C'+str(opt)] = -self.__dict__['C'+str(opt)]
            
            self._strike_tenor_label()
 
        if self.y_plot == 'rho':
            self.T1 = self.T
            self.T2 = self.T + self.time_shift
            tenor_type = {1:1, 2:2, 3:1, 4:2}
            opt_type = {1:'call', 2:'call', 3:'put', 4:'put'}
            for opt in [1, 2, 3, 4]:
                if self.x_plot == 'price':
                    self.__dict__['C'+str(opt)] = getattr(self, str(self.y_plot))(S=self.SA, K=self.G2, T=self.__dict__['T'+str(tenor_type[opt])], 
                                                                             r=self.r, q=self.q, sigma=self.sigma, option=opt_type[opt], refresh='graph')
                if self.x_plot == 'strike':
                    self.__dict__['C'+str(opt)] = getattr(self, str(self.y_plot))(S=self.S0, K=self.SA, T=self.__dict__['T'+str(tenor_type[opt])],
                                                                             r=self.r, q=self.q, sigma=self.sigma, option=opt_type[opt], refresh='graph')
                if self.x_plot == 'vol':
                    self.__dict__['C'+str(opt)] = getattr(self, str(self.y_plot))(S=self.S0, K=self.G2, T=self.__dict__['T'+str(tenor_type[opt])], 
                                                                             r=self.r, sigma=self.sigmaA, option=opt_type[opt], refresh='graph')
                    
            if self.direction == 'short':
                for opt in [1, 2, 3, 4]:
                    self.__dict__['C'+str(opt)] = -self.__dict__['C'+str(opt)]
    
            self.label1 = str(int(self.T1*365))+' Day Call'
            self.label2 = str(int(self.T2*365))+' Day Call'
            self.label3 = str(int(self.T1*365))+' Day Put'
            self.label4 = str(int(self.T2*365))+' Day Put'
    
    
        self.xlabel = self.label_dict[str(self.x_plot)]
        self.ylabel = self.label_dict[str(self.y_plot)]
        
        if self.y_plot in [self.equal_greeks, 'rho']:
                self.option = 'Call / Put'     
            
        self.title = (str(self.direction.title())+' '+str(self.option.title())+
                      ' '+self.y_plot.title()+' vs '+self.x_plot.title())   
            
        self.x_name = str(self.x_plot)
        if self.x_name in self.x_name_dict.keys():
            self.xarray = (self.__dict__[str(self.x_name_dict[self.x_name])] * 
                           self.x_scale_dict[self.x_name])

        if self.y_plot in self.y_name_dict.keys():        
            self._vis_greeks_mpl(yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                                 xarray1=self.xarray, xarray2=self.xarray, xarray3=self.xarray, 
                                 label1=self.label1, label2=self.label2, label3=self.label3, 
                                 xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)       
    
        if self.y_plot == 'rho':
            self._vis_greeks_mpl(yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                                 yarray4=self.C4, xarray1=self.xarray, xarray2=self.xarray, 
                                 xarray3=self.xarray, xarray4=self.xarray, label1=self.label1, 
                                 label2=self.label2, label3=self.label3, label4=self.label4, 
                                 xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
 
    
    def payoffs(self, S0=None, K=None, K1=None, K2=None, K3=None, K4=None, 
                T=None, r=None, q=None, sigma=None, option=None, direction=None, 
                cash=None, ratio=None, value=None, combo_payoff=None):
        """
        Displays the graph of the specified combo payoff.
                
        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of option 1. The default is 100 (individual payoffs 
                                                          may have own defaults).
        K1 : Float
             Strike Price of option 1. The default is 95 (individual payoffs 
                                                          may have own defaults).
        K2 : Float
             Strike Price of option 2. The default is 105 (individual payoffs 
                                                          may have own defaults).
        K3 : Float
             Strike Price of option 3. The default is 105 (individual payoffs 
                                                          may have own defaults).
        K4 : Float
             Strike Price of option 4. The default is 105 (individual payoffs 
                                                          may have own defaults).
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        cash : Bool
            Whether to discount forward to present value. The default is False.
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. The default is 2. 
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.
        combo_payoff : Str
            The payoff to be displayed.

        Returns
        -------
        Runs the specified combo payoff method.

        """
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
                            ratio=ratio, value=value)
        
        if combo_payoff == 'ratio vertical spread':
            self.ratio_vertical_spread(S0=S0, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                                       option=option, ratio=ratio, value=value)
        
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
            
    
    def call(self, S0=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, 
             value=None):
        """
        Displays the graph of the call.

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of option 1. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The 
            default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'call'
        self.option1 = 'call'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                              value=value, option1=self.option1)
        
        self._return_options(legs=1)
        
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
                
        
    def put(self, S0=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, 
            value=None):
        """
        Displays the graph of the put.

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of option 1. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'put'
        self.option1 = 'put'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                              value=value, option1=self.option1)
        
        self._return_options(legs=1)
        
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
        """
        Displays the graph of the underlying.

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        direction : Str
            Whether the payoff is long or short. The default is 'long'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'stock'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, direction=direction)
        
        self.SA = np.linspace(0.8 * self.S0, 1.2 * self.S0, 100)
        
        if self.direction == 'long':
            payoff = self.SA - self.S0
            title = 'Long Stock'
        if self.direction == 'short':
            payoff = self.S0 - self.SA
            title = 'Short Stock'
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, label='Payoff', 
                         title=title)     
            
    
    def forward(self, S0=None, T=None, r=None, q=None, sigma=None, direction=None, 
                cash=None):
        """
        Displays the graph of the synthetic forward strategy:
            Long one ATM call
            Short one ATM put

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        cash : Bool
            Whether to discount to present value. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'forward'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, T=T, r=r, q=q, sigma=sigma, option1='call', 
                              option2='put', direction=direction, cash=cash)
        
        self.K1 = self.S0
        self.K2 = self.S0
        self.T1 = self.T
        self.T2 = self.T
        
        self._return_options(legs=2)
        
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
        
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, label='Payoff', 
                         title=title)
    
    
    def collar(self, S0=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
               direction=None, value=None):
        """
        Displays the graph of the collar strategy:
            Long one OTM put
            Short one OTM call

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 98.
        K2 : Float
             Strike Price of option 2. The default is 102.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'collar'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma,
                              option1='put', option2='call', direction=direction, 
                              value=value)

        self._return_options(legs=2)
        
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
        """
        Displays the graph of the spread strategy:
            Long one ITM option
            Short one OTM option

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.


        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'spread'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option=option, option1=option, option2=option, direction=direction, 
                              value=value)
        
        self._return_options(legs=2)
 
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
        """
        Displays the graph of the backspread strategy:
            Short one ITM option
            Long ratio * OTM options

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. The default is 2.    
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'backspread'

        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option=option, option1=option, option2=option, ratio=ratio, 
                              value=value)
        
        self._return_options(legs=2)
        
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
        """
        Displays the graph of the ratio vertical spread strategy:
            Long one ITM option
            Short ratio * OTM options

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. The default is 2.    
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.
        
        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'ratio vertical spread'

        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option=option, option1=option, option2=option, ratio=ratio, 
                              value=value)
        
        self._return_options(legs=2)
        
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
        """
        Displays the graph of the straddle strategy:
            Long one ATM put
            Long one ATM call

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of options 1 and 2. The default is 100.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.
        
        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.    

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'straddle'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K=K, K1=K, K2=K, T=T, T1=T, T2=T, r=r, q=q, 
                              sigma=sigma, option1='put', option2='call', direction=direction, 
                              value=value)
        
        self._return_options(legs=2)
        
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
        """
        Displays the graph of the strangle strategy:
            Long one OTM put
            Long one OTM call

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.
        
        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'strangle'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option1='put', option2='call', direction=direction, 
                              value=value)
        
        self._return_options(legs=2)
        
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
        """
        Displays the graph of the butterfly strategy:
            Long one ITM option
            Short two ATM options
            Long one OTM option

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'butterfly'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, T=T, T1=T, T2=T, 
                              T3=T, r=r, q=q, sigma=sigma, option=option, option1=option, 
                              option2=option, option3=option, direction=direction, 
                              value=value)
        
        self._return_options(legs=3)
              
        if self.direction == 'long':
            payoff = (self.C1 - 2*self.C2 + self.C3 - self.C1_0 + 2*self.C2_0 - self.C3_0)
            if self.value == True:
                payoff2 = (self.C1_G - 2*self.C2_G + self.C3_G - self.C1_0 + 2*self.C2_0 - self.C3_0)
            else:
                payoff2 = None
                
        elif self.direction == 'short':    
            payoff = (-self.C1 + 2*self.C2 - self.C3 + self.C1_0 - 2*self.C2_0 + self.C3_0)
            if self.value == True:
                payoff2 = (-self.C1_G + 2*self.C2_G - self.C3_G + self.C1_0 - 2*self.C2_0 + self.C3_0)
            else:
                payoff2 = None
        
        else:         
            print('Check Inputs')
                
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
        """
        Displays the graph of the christmas tree strategy:
            Long one ITM option
            Short one ATM option
            Short one OTM option

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.
        
        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'christmas tree'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, T=T, T1=T, T2=T, 
                              T3=T, r=r, q=q, sigma=sigma, option=option, option1=option, 
                              option2=option, option3=option, direction=direction, 
                              value=value)
        
        self._return_options(legs=3)
        
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


    def condor(self, S0=None, K1=None, K2=None, K3=None, K4=None, T=None, r=None, 
               q=None, sigma=None, option=None, direction=None, value=None):
        """
        Displays the graph of the condor strategy:
            Long one low strike option
            Short one option with a higher strike
            Short one option with a higher strike 
            Long one option with a higher strike        

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 90.
        K2 : Float
             Strike Price of option 2. The default is 95.
        K3 : Float
             Strike Price of option 3. The default is 100.
        K4 : Float
             Strike Price of option 4. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults        
        self.combo_payoff = 'condor'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values 
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T=T, T1=T, T2=T, 
                              T3=T, T4=T, r=r, q=q, sigma=sigma, option=option, option1=option, 
                              option2=option, option3=option, option4=option, direction=direction, 
                              value=value)
        
        self._return_options(legs=4)
        
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


    def iron_butterfly(self, S0=None, K1=None, K2=None, K3=None, K4=None, T=None, r=None, 
                       q=None, sigma=None, direction=None, value=None):
        """
        Displays the graph of the iron butterfly strategy:
            Short one OTM put
            Long one ATM put
            Long one ATM call 
            Short one OTM call
        Akin to having a long straddle inside a larger short strangle (or vice-versa)

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 100.
        K4 : Float
             Strike Price of option 4. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'iron butterfly'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T=T, T1=T, T2=T, 
                              T3=T, T4=T, r=r, q=q, sigma=sigma, option1='put', 
                              option2='call', option3='put', option4='call', direction=direction, 
                              value=value)
        
        self._return_options(legs=4)
        
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
                       q=None, sigma=None, direction=None, value=None):
        """
        Displays the graph of the iron condor strategy:
            Long one OTM put
            Short one OTM put with a higher strike
            Short one OTM call 
            Long one OTM call with a higher strike
        Akin to having a long strangle inside a larger short strangle (or vice-versa)   

        Parameters
        ----------
        S0 : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 90.
        K2 : Float
             Strike Price of option 2. The default is 95.
        K3 : Float
             Strike Price of option 3. The default is 100.
        K4 : Float
             Strike Price of option 4. The default is 105.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal payoff. The default is False.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation takes into account specific defaults
        self.combo_payoff = 'iron condor'
        
        # Pass parameters to be initialised. If not provided they will be populated with default values 
        self._initialise_func(S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, T=T, T1=T, T2=T, 
                              T3=T, T4=T, r=r, q=q, sigma=sigma, option1='put', 
                              option2='put', option3='call', option4='call', direction=direction, 
                              value=value)
        
        self._return_options(legs=4)
        
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
                
        if self.direction == 'long':
            title = 'Long Iron Condor'
        if self.direction == 'short':
            title = 'Short Iron Condor'
       
        self._vis_payoff(S0=self.S0, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
    
    
    
    def _return_options(self, legs=2):
        """
        Calculate option prices to be used in payoff diagrams.

        Parameters
        ----------
        legs : Int
            Number of option legs to calculate. The default is 2.

        Returns
        -------
        From 1 to 4 sets of option values:
            Cx_0: Current option price; Float.
            Cx: Terminal Option payoff, varying by strike; Array
            Cx_G: Current option value, varying by strike; Array

        """
        
        # array of 1000 equally spaced points between 75% of initial underlying price and 125%
        self.SA = np.linspace(0.75 * self.S0, 1.25 * self.S0, 1000)
                
        self.C1_0 = self.price(S=self.S0, K=self.K1, T=self.T1, r=self.r, q=self.q, 
                               sigma=self.sigma, option=self.option1, refresh='graph')
        self.C1 = self.price(S=self.SA, K=self.K1, T=0, r=self.r, q=self.q, sigma=self.sigma, 
                             option=self.option1, refresh='graph')
        self.C1_G = self.price(S=self.SA, K=self.K1, T=self.T1, r=self.r, q=self.q, 
                               sigma=self.sigma, option=self.option1, refresh='graph')
        
        if legs > 1:
            self.C2_0 = self.price(S=self.S0, K=self.K2, T=self.T2, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option2, refresh='graph')
            self.C2 = self.price(S=self.SA, K=self.K2, T=0, r=self.r, q=self.q, sigma=self.sigma, 
                                 option=self.option2, refresh='graph')
            self.C2_G = self.price(S=self.SA, K=self.K2, T=self.T2, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option2, refresh='graph')

        if legs > 2:
            self.C3_0 = self.price(S=self.S0, K=self.K3, T=self.T3, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option3, refresh='graph')
            self.C3 = self.price(S=self.SA, K=self.K3, T=0, r=self.r, q=self.q, 
                                 sigma=self.sigma, option=self.option3, refresh='graph')
            self.C3_G = self.price(S=self.SA, K=self.K3, T=self.T3, r=self.r, q=self.q, sigma=self.sigma, 
                                   option=self.option3, refresh='graph')
        
        if legs > 3:
            self.C4_0 = self.price(S=self.S0, K=self.K4, T=self.T4, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option4, refresh='graph')
            self.C4 = self.price(S=self.SA, K=self.K4, T=0, r=self.r, q=self.q, sigma=self.sigma, 
                                 option=self.option4, refresh='graph')
            self.C4_G = self.price(S=self.SA, K=self.K4, T=self.T4, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option4, refresh='graph')
        
        return self
        
    



        
        
        