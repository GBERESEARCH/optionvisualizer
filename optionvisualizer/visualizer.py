import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot

# Dictionary of default parameters
df_dict = {'df_S':100, 
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
           'df_r':0.005,
           'df_b':0.005,
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
           'df_price_shift':0.25,
           'df_price_shift_type':'avg',
           'df_vol_shift':0.001,           
           'df_ttm_shift':1/365,
           'df_rate_shift':0.0001,
           'df_greek':'delta',
           'df_num_sens':False,
           'df_interactive':False,
           'df_notebook':True,
           'df_colorscheme':'jet',
           'df_colorintensity':1,
           'df_size':(15, 12),
           'df_graphtype':'2D',
           'df_y_plot':'delta',
           'df_x_plot':'time',
           'df_time_shift':0.25,
           'df_cash':False,
           'df_axis':'price',
           'df_spacegrain':100,
           'df_risk':True,

            # List of default parameters used when refreshing 
            'df_params_list':['S', 'K', 'K1', 'K2', 'K3', 'K4', 'G1', 'G2', 'G3', 
                              'H', 'R', 'T', 'T1', 'T2', 'T3', 'T4', 'r', 'b', 'q', 
                              'sigma', 'eta', 'phi', 'barrier_direction', 'knock', 
                              'option', 'option1', 'option2', 'option3', 'option4', 
                              'direction', 'value', 'ratio', 'refresh', 'price_shift', 
                              'price_shift_type', 'vol_shift','ttm_shift', 'rate_shift', 
                              'greek', 'num_sens', 'interactive', 'notebook', 'colorscheme', 
                              'colorintensity', 'size', 'graphtype', 'cash', 'axis', 
                              'spacegrain', 'risk'],
            
            # List of Greeks where call and put values are the same
            'df_equal_greeks':['gamma', 'vega', 'vomma', 'vanna', 'zomma', 'speed', 
                              'color', 'ultima', 'vega bleed'],
            
            # Payoffs requiring changes to default parameters
            'df_mod_payoffs':['call', 'put', 'collar', 'straddle', 'butterfly', 
                              'christmas tree', 'condor', 'iron butterfly', 'iron condor'],
            
            # Those parameters that need changing
            'df_mod_params':['S', 'K', 'K1', 'K2', 'K3', 'K4', 'T', 'T1', 'T2', 
                             'T3', 'T4'],
            
            # Combo parameter values differing from standard defaults
            'df_combo_dict':{'call':{'S':100,
                                     'K':100,
                                     'K1':100,
                                     'T1':0.25},
                             'put':{'S':100,
                                    'K':100,
                                    'K1':100,
                                    'T1':0.25},
                             'collar':{'S':100,
                                       'K':100,
                                       'K1':98,
                                       'K2':102,
                                       'T1':0.25,
                                       'T2':0.25},
                             'straddle':{'S':100,
                                         'K':100,
                                         'K1':100,
                                         'K2':100,
                                         'T1':0.25,
                                         'T2':0.25},
                             'butterfly':{'S':100,
                                          'K':100,
                                          'K1':95,
                                          'K2':100,
                                          'K3':105,
                                          'T1':0.25,
                                          'T2':0.25,
                                          'T3':0.25},
                             'christmas tree':{'S':100,
                                               'K':100,
                                               'K1':95,
                                               'K2':100,
                                               'K3':105,
                                               'T1':0.25,
                                               'T2':0.25,
                                               'T3':0.25},
                             'condor':{'S':100,
                                       'K':100,
                                       'K1':90,
                                       'K2':95,
                                       'K3':100,
                                       'K4':105,
                                       'T1':0.25,
                                       'T2':0.25,
                                       'T3':0.25,
                                       'T4':0.25},
                             'iron butterfly':{'S':100,
                                               'K':100,
                                               'K1':95,
                                               'K2':100,
                                               'K3':100,
                                               'K4':105,
                                               'T1':0.25,
                                               'T2':0.25,
                                               'T3':0.25,
                                               'T4':0.25},
                             'iron condor':{'S':100,
                                            'K':100,
                                            'K1':90,
                                            'K2':95,
                                            'K3':100,
                                            'K4':105,
                                            'T1':0.25,
                                            'T2':0.25,
                                            'T3':0.25,
                                            'T4':0.25}},
            
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
    
    def __init__(self, S=df_dict['df_S'], K=df_dict['df_K'], 
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
                 combo_payoff=df_dict['df_combo_payoff'], price_shift=df_dict['df_price_shift'], 
                 price_shift_type=df_dict['df_price_shift_type'], vol_shift=df_dict['df_vol_shift'], 
                 ttm_shift=df_dict['df_ttm_shift'], rate_shift=df_dict['df_rate_shift'], 
                 greek=df_dict['df_greek'], num_sens=df_dict['df_num_sens'],
                 interactive=df_dict['df_interactive'], notebook=df_dict['df_notebook'], 
                 colorscheme=df_dict['df_colorscheme'], colorintensity=df_dict['df_colorintensity'], 
                 size=df_dict['df_size'], graphtype=df_dict['df_graphtype'], y_plot=df_dict['df_y_plot'], 
                 x_plot=df_dict['df_x_plot'], x_name_dict=df_dict['df_x_name_dict'], 
                 x_scale_dict=df_dict['df_x_scale_dict'], y_name_dict=df_dict['df_y_name_dict'], 
                 time_shift=df_dict['df_time_shift'], cash=df_dict['df_cash'], axis=df_dict['df_axis'], 
                 spacegrain=df_dict['df_spacegrain'], risk=df_dict['df_risk'], df_combo_dict=df_dict['df_combo_dict'], 
                 df_params_list=df_dict['df_params_list'], equal_greeks=df_dict['df_equal_greeks'], 
                 mod_payoffs=df_dict['df_mod_payoffs'], mod_params=df_dict['df_mod_params'], 
                 label_dict=df_dict['df_label_dict'], greek_dict=df_dict['df_greek_dict'], 
                 df_dict=df_dict):

        self.S = S # Spot price
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
        self.price_shift = price_shift # Size of price shift used in shift_greeks function
        self.price_shift_type = price_shift_type # Shift type - Up, Down or Avg
        self.vol_shift = vol_shift # Size of vol shift used in shift_greeks function
        self.ttm_shift = ttm_shift # Size of time shift used in shift_greeks function
        self.rate_shift = rate_shift # Size of interest rate shift used in shift_greeks function
        self.df_dict = df_dict # Dictionary of parameter defaults
        self.df_combo_dict = df_combo_dict # Dictionary of payoffs with different default parameters
        self.df_params_list = df_params_list # List of default parameters
        self.greek = greek # Option greek to display e.g. delta
        self.num_sens = num_sens # Whether to calculate numerical or analytical sensitivity
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
        self.spacegrain = spacegrain # Number of points in each axis linspace argument for 3D graphs
        self.risk = risk # Whether to show risk or payoff graphs in visualize method
        self.mod_payoffs = mod_payoffs # Combo payoffs needing different default parameters
        self.mod_params = mod_params # Parameters of these payoffs that need changing
        self.label_dict = label_dict # Dictionary mapping function parameters to axis labels
        self.equal_greeks = equal_greeks # List of Greeks where call and put values are the same
        self.greek_dict = greek_dict # Greek names as function input and individual function names
        self.combo_payoff = combo_payoff # 2D graph payoff structure

    
    def _initialise_func(self, **kwargs):
        """
        Initialise pricing data.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if no data provided and recalculate 
            distributions based on updated data.

        """
        self._refresh_params(**kwargs)
        self._refresh_dist()
        
        return self


    def _initialise_graphs(self, **kwargs):
        """
        Initialise pricing data for graphs.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters (resetting defaults will have taken 
            place earlier in the process) and recalculate distributions based on updated data.

        """
        self._set_params(**kwargs)
        self._refresh_dist()
        
        return self
    

    def _initialise_barriers(self, **kwargs):
        """
        Initialise pricing data for graphs.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if no data provided, calculate 
            distributions based on updated data and calculate the barrier option specific parameters.

        """
        self._refresh_params(**kwargs)
        self._refresh_dist()
        self._barrier_factors()

        return self


    def _refresh_params(self, **kwargs):
        """
        Set parameters for use in various pricing functions

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if no data provided

        """
        
        # Certain combo payoffs (found in the mod_payoffs list) require specific default parameters
        if self.combo_payoff in self.mod_payoffs:
            for k, v in kwargs.items():
                if v is None:
                    
                    # These parameters are in the mod_params list
                    if k in self.mod_params:
                        try:
                            
                            # Extract these from the df_combo_dict
                            v = self.df_combo_dict[str(self.combo_payoff)][str(k)]
                        except:
                            
                            # Otherwise set to the standard default value
                            v = df_dict['df_'+str(k)]
                    if k not in self.mod_params:
                        v = df_dict['df_'+str(k)]
                    
                    # Now assign this to the object
                    self.__dict__[k] = v
                
                # If the parameter has been provided as an input, assign this to the object
                else:
                    self.__dict__[k] = v
           
        else:
            
            # For all the other combo_payoffs
            for k, v in kwargs.items():
                
                # If a parameter has not been provided
                if v is None:
                    
                    # Set it to the default value and assign to the object
                    v = df_dict['df_'+str(k)]
                    self.__dict__[k] = v
                
                # If the parameter has been provided as an input, assign this to the object
                else:
                    self.__dict__[k] = v
        
        # For each parameter in the list of parameters to be updated that was not supplied as a kwarg 
        for key in list(set(self.df_params_list) - set(kwargs.keys())):
            if key not in kwargs:
                
                # Set it to the default value and assign to the object
                val = df_dict['df_'+str(key)]
                self.__dict__[key] = val
                
        return self        
   
    
    def _set_params(self, **kwargs):
        """
        Fix parameters for use in various pricing functions

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods that use it to refresh data.

        Returns
        -------
        Various
            Assigns input parameters to the object

        """
        
        # For each input parameter provided
        for k, v in kwargs.items():
            if v is not None:
                # Assign to the object
                self.__dict__[k] = v
    
        return self
       
        
    def _refresh_dist(self):
        """
        Calculate various parameters and distributions

        Returns
        -------
        Various
            Assigns parameters to the object

        """
        
        # Cost of carry as risk free rate less dividend yield
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
        Calculate the barrier option specific parameters

        Returns
        -------
        Various
            Assigns parameters to the object

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
        Black Scholes Option Price

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
            Black Scholes Option Price.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
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
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Delta. 

        """    
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
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
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Theta.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
                   
        if self.option == 'call':
            self.opt_theta = ((-self.S * self.carry * self.nd1 * self.sigma ) / 
                              (2 * np.sqrt(self.T)) - (self.b - self.r) * self.S * self.carry * 
                              self.Nd1 - self.r * self.K * np.exp(-self.r * self.T) * self.Nd2) / 100 
        if self.option == 'put':   
            self.opt_theta = ((-self.S * self.carry * self.nd1 * self.sigma ) / 
                              (2 * np.sqrt(self.T)) + (self.b - self.r) * self.S * self.carry * 
                              self.minusNd1 + self.r * self.K * np.exp(-self.r * self.T) * self.minusNd2) / 100

        return self.opt_theta
    
    
    def gamma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        Sensitivity of delta to changes in the underlying asset price

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Gamma.

        """
               
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
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
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Vega.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)

        self.opt_vega = (self.S * self.carry * self.nd1 * np.sqrt(self.T)) / 100
        
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
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Rho.

        """        
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
        
        if self.option == 'call':
            self.opt_rho = (self.T * self.K * np.exp(-self.r * self.T) * self.Nd2) / 10000
        if self.option == 'put':
            self.opt_rho = (-self.T * self.K * np.exp(-self.r * self.T) * self.minusNd2) / 10000
            
        return self.opt_rho


    def vanna(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, refresh=None):
        """
        DdeltaDvol, DvegaDspot 
        Sensitivity of delta to changes in volatility
        Sensitivity of vega to changes in the asset price   

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Vanna.

        """
              
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_vanna = (((-self.carry * self.d2) / self.sigma) * self.nd1) / 100 

        return self.opt_vanna               
 

    def vomma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        DvegaDvol, Vega Convexity, Volga, Vol Gamma
        Sensitivity of vega to changes in volatility

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Vomma.

        """
               
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_vomma = (self.vega(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                    self.option, self.refresh) * ((self.d1 * self.d2) / (self.sigma))) / 100
        
        return self.opt_vomma


    def charm(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
              refresh=None):
        """
        DdeltaDtime, Delta Bleed 
        Sensitivity of delta to changes in time to maturity

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Charm.

        """
               
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    refresh=refresh)
        
        if self.option == 'call':
            self.opt_charm = (-self.carry * ((self.nd1 * ((self.b / (self.sigma * np.sqrt(self.T))) - 
                                                          (self.d2 / (2 * self.T)))) + 
                                             ((self.b - self.r) * self.Nd1))) / 100
        if self.option == 'put':
            self.opt_charm = (-self.carry * ((self.nd1 * ((self.b / (self.sigma * np.sqrt(self.T))) - 
                                                          (self.d2 / (2 * self.T)))) - 
                                             ((self.b - self.r) * self.minusNd1))) / 100
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
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Zomma.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_zomma = (self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                     self.option, self.refresh) * ((self.d1 * self.d2 - 1) / self.sigma)) / 100
        
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
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Speed.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
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
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Color.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_color = (self.gamma(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                     self.option, self.refresh) * ((self.r - self.b) + 
                                    ((self.b * self.d1) / (self.sigma * np.sqrt(self.T))) + 
                                    ((1 - self.d1 * self.d2) / (2 * self.T)))) / 100
        
        return self.opt_color


    def ultima(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
               refresh=None):
        """
        DvommaDvol
        Sensitivity of vomma to changes in volatility
        3rd derivative of option price wrt volatility

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Ultima.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_ultima = (self.vomma(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                                      self.option, self.refresh) * ((1 / self.sigma) * 
                                     (self.d1 * self.d2 - (self.d1 / self.d2) - 
                                     (self.d2 / self.d1) - 1))) / 100
        
        return self.opt_ultima


    def vega_bleed(self, S=None, K=None, T=None, r=None, q=None, sigma=None, option=None, 
                   refresh=None):
        """
        DvegaDtime
        Sensitivity of vega to changes in time to maturity.

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
            Option Vega Bleed.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, refresh=refresh)
        
        self.opt_vega_bleed = (self.vega(self.S, self.K, self.T, self.r, self.q, self.sigma, 
                               self.option, self.refresh) * (self.r - self.b + 
                             ((self.b * self.d1) / (self.sigma * np.sqrt(self.T))) - 
                              ((1 + (self.d1 * self.d2) ) / (2 * self.T)))) / 100

        return self.opt_vega_bleed


    def analytical_sensitivities(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                                 option=None, greek=None, refresh=None):
        """
        Sensitivities of the option calculated analytically from closed form solutions.
        
        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        greek : Str
            Sensitivity to return. Select from 'delta', 'gamma', 'vega', 'theta',
            'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 'ultima', 'vega bleed',
            'charm'. The default is 'delta'            
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        Float
            Option Sensitivity.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                  greek=greek)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, greek=greek, 
                                    option=option, refresh=refresh)
        
        for key, value in self.greek_dict.items():
            if str(self.greek) == key:
                return getattr(self, value)(S=self.S, K=self.K, T=self.T, r=self.r, 
                                            q=self.q, sigma=self.sigma, option=self.option, refresh='graph')
                


    def numerical_delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                        option=None, price_shift=None, price_shift_type=None, refresh=None):
        """
        Sensitivity of the option price to changes in asset price
        Calculated by taking the difference in price for varying shift sizes

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        price_shift : Float
            The size of the up and down shift in basis points. The default is 25.
        price_shift_type : Str
            Whether to calculate the change for an upshift, downshift or average of the two. The default is 'avg'.
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 

        Returns
        -------
        Float
            Option Delta.

        """
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                  price_shift=price_shift, price_shift_type=price_shift_type)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option,
                                    price_shift=price_shift, price_shift_type=price_shift_type, 
                                    refresh=refresh)
        
        down_shift = self.S-(self.price_shift/10000)*self.S
        up_shift = self.S+(self.price_shift/10000)*self.S
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
    
    
    def numerical_sensitivities(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                                option=None, greek=None, price_shift=None, vol_shift=None, 
                                ttm_shift=None, refresh=None):
        """
        Sensitivities of the option calculated numerically using shifts in parameters.

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        greek : Str
            Sensitivity to return. Select from 'delta', 'gamma', 'vega', 'theta',
            'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 'ultima', 'vega bleed',
            'charm'. The default is 'delta'
        price_shift : Float
            The size of the price shift in decimal terms. The default is 0.25.
        vol_shift : Float
            The size of the volatility shift in decimal terms. The default is 0.001.
        ttm_shift : Float
            The size of the time to maturity shift in decimal terms. The default is 1/365.    
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 

        Returns
        -------
        Float
            Option Sensitivity.

        """
        
        
        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                  greek=greek, price_shift=price_shift, vol_shift=vol_shift, 
                                  ttm_shift=ttm_shift)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    greek=greek, price_shift=price_shift, vol_shift=vol_shift, 
                                    ttm_shift=ttm_shift, refresh=refresh)
        
        # Store initial values
        self.S0 = self.S
        self.sigma0 = self.sigma
        self.T0 = self.T
        self.r0 = self.r
        self.q0 = self.q
        
        if self.greek == 'delta':
            result = ((self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      self.price(S=(self.S0 - self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (2 * self.price_shift))
        
        if self.greek == 'gamma':
            result = ((self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      (2 * self.price(S=self.S0, K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) +
                       self.price(S=(self.S0 - self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (self.price_shift ** 2))
            
        if self.greek == 'vega':
            result = ((self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0+self.vol_shift), option=self.option, refresh='graph') - 
                      self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                 sigma=(self.sigma0-self.vol_shift), option=self.option, refresh='graph')) / 
                      (2 * self.vol_shift)) / 100
        
        if self.greek == 'theta':
            result = ((self.price(S=self.S0, K=self.K, T=(self.T0-self.ttm_shift), r=self.r0, 
                                  q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, 
                                 q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (self.ttm_shift * 100)) 
        
        if self.greek == 'rho':
            result = ((self.price(S=self.S0, K=self.K, T=self.T0, r=(self.r0+self.rate_shift), 
                                  q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      self.price(S=self.S0, K=self.K, T=self.T0, r=(self.r0-self.rate_shift), 
                                 q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (2 * self.rate_shift * 10000))
                      
        if self.greek == 'vomma':
            result = ((self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0+self.vol_shift), option=self.option, refresh='graph') - 
                      (2 * self.price(S=self.S0, K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) +
                       self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0-self.vol_shift), option=self.option, refresh='graph')) / 
                      (self.vol_shift ** 2)) / 10000              
        
        if self.greek == 'vanna':
            result = ((1 / (4 * self.price_shift * self.vol_shift)) *
                      (self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0+self.vol_shift), option=self.option, refresh='graph') - 
                      self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                 sigma=(self.sigma0-self.vol_shift), option=self.option, refresh='graph') - 
                      self.price(S=(self.S0 - self.price_shift), K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0+self.vol_shift), option=self.option, refresh='graph') + 
                      self.price(S=(self.S0 - self.price_shift), K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                 sigma=(self.sigma0-self.vol_shift), option=self.option, refresh='graph'))) / 100
        
        if self.greek == 'charm':
            result = (((self.price(S=(self.S0 + self.price_shift), K=self.K, T=(self.T0-self.ttm_shift), 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      self.price(S=(self.S0 - self.price_shift), K=self.K, T=(self.T0-self.ttm_shift), 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (2 * self.price_shift)) -
                      ((self.price(S=(self.S0 + self.price_shift), K=self.K, T=(self.T0), 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      self.price(S=(self.S0 - self.price_shift), K=self.K, T=(self.T0), 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (2 * self.price_shift))) / (self.ttm_shift * 100)
        
        if self.greek == 'zomma':
            result = (((self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=(self.sigma0+self.vol_shift), 
                                 option=self.option, refresh='graph') - 
                      (2 * self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                      sigma=(self.sigma0+self.vol_shift), option=self.option, 
                                      refresh='graph')) +
                       self.price(S=(self.S0 - self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=(self.sigma0+self.vol_shift), 
                                 option=self.option, refresh='graph')) -
                      self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=(self.sigma0-self.vol_shift), 
                                 option=self.option, refresh='graph') + 
                      (2 * self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                      sigma=(self.sigma0-self.vol_shift), option=self.option, 
                                      refresh='graph')) -
                       self.price(S=(self.S0 - self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=(self.sigma0-self.vol_shift), 
                                 option=self.option, refresh='graph')) / 
                      (2 * self.vol_shift * (self.price_shift ** 2)) / 100)
        
        if self.greek == 'speed':
            result = (1 / (self.price_shift ** 3) *
                      (self.price(S=(self.S0 + (2 * self.price_shift)), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      (3 * self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) +
                       3 * self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                      sigma=self.sigma0, option=self.option, refresh='graph') -
                       self.price(S=(self.S0-self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')))
                
        if self.greek == 'color':
            result = (((self.price(S=(self.S0 + self.price_shift), K=self.K, T=(self.T0-self.ttm_shift), 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      (2 * self.price(S=self.S0, K=self.K, T=(self.T0-self.ttm_shift), 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) +
                       self.price(S=(self.S0 - self.price_shift), K=self.K, T=(self.T0-self.ttm_shift), 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (self.price_shift ** 2)) - 
                      ((self.price(S=(self.S0 + self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph') - 
                      (2 * self.price(S=self.S0, K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) +
                       self.price(S=(self.S0 - self.price_shift), K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=self.sigma0, option=self.option, refresh='graph')) / 
                      (self.price_shift ** 2) )) / (self.ttm_shift * 100)
            
        if self.greek == 'ultima':
            result = (1 / (self.vol_shift ** 3) *
                      (self.price(S=self.S0, K=self.K, T=self.T0, 
                                 r=self.r0, q=self.q0, sigma=(self.sigma0 + (2 * self.vol_shift)),
                                 option=self.option, refresh='graph') - 
                      (3 * self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                      sigma=(self.sigma0 + self.vol_shift), option=self.option, 
                                      refresh='graph')) +
                       3 * self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                      sigma=self.sigma0, option=self.option, refresh='graph') -
                       self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0 - self.vol_shift), option=self.option, 
                                  refresh='graph'))) * (self.vol_shift ** 2)
        
        if self.greek == 'vega bleed':
            result = (((self.price(S=self.S0, K=self.K, T=(self.T0-self.ttm_shift), r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0+self.vol_shift), option=self.option, refresh='graph') - 
                      self.price(S=self.S0, K=self.K, T=(self.T0-self.ttm_shift), r=self.r0, q=self.q0, 
                                 sigma=(self.sigma0-self.vol_shift), option=self.option, refresh='graph')) / 
                      (2 * self.vol_shift)) - 
                      ((self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                  sigma=(self.sigma0+self.vol_shift), option=self.option, refresh='graph') - 
                      self.price(S=self.S0, K=self.K, T=self.T0, r=self.r0, q=self.q0, 
                                 sigma=(self.sigma0-self.vol_shift), option=self.option, refresh='graph')) / 
                      (2 * self.vol_shift))) / (self.ttm_shift * 10000)
        
        return result


    def sensitivities(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                      option=None, greek=None, price_shift=None, vol_shift=None, 
                      ttm_shift=None, num_sens=None, refresh=None):
        """
        Sensitivities of the option.

        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        greek : Str
            Sensitivity to return. Select from 'delta', 'gamma', 'vega', 'theta',
            'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 'ultima', 'vega bleed',
            'charm'. The default is 'delta'
        price_shift : Float
            The size of the price shift in decimal terms. The default is 0.25.
        vol_shift : Float
            The size of the volatility shift in decimal terms. The default is 0.001.
        ttm_shift : Float
            The size of the time to maturity shift in decimal terms. The default is 1/365. 
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. The default is False.
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place. 

        Returns
        -------
        Float
            Option Sensitivity.

        """

        # If refresh is set to 'graph' the price is to be used in combo graphs so 
        # the distributions are refreshed but not the parameters.
        if refresh == 'Std' or refresh is None:
            self._initialise_func(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                  greek=greek, price_shift=price_shift, vol_shift=vol_shift, 
                                  ttm_shift=ttm_shift, num_sens=num_sens)
        if refresh == 'graph':
            self._initialise_graphs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                                    greek=greek, price_shift=price_shift, vol_shift=vol_shift, 
                                    ttm_shift=ttm_shift, num_sens=num_sens, refresh=refresh)
            
        if self.num_sens == False:
            return self.analytical_sensitivities(S=self.S, K=self.K, T=self.T, r=self.r, 
                                                 q=self.q, sigma=self.sigma, option=self.option, 
                                                 greek=self.greek, refresh=self.refresh)
        else:
            return self.numerical_sensitivities(S=self.S, K=self.K, T=self.T, r=self.r, 
                                                q=self.q, sigma=self.sigma, option=self.option, 
                                                greek=self.greek, price_shift=self.price_shift, 
                                                vol_shift=self.vol_shift, ttm_shift=self.ttm_shift, 
                                                refresh=self.refresh)


    def barrier_price(self, S=None, K=None, H=None, R=None, T=None, r=None, q=None, 
                       sigma=None, barrier_direction=None, knock=None, option=None, 
                       refresh=None):
        """
        Return the Barrier option price
    
        Parameters
        ----------
        S : Float
            Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price. The default is 100.
        H : Float
            Barrier Level. The default is 105.
        R : Float
            Rebate. The default is 0.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        barrier_direction : Str
            Up or Down. The default is 'up'.
        knock : Str
            knock-in or knock-out. The default is 'in'.
        option : Str
            Option type, Put or Call. The default is 'call'
        refresh : Str
            Whether the function is being called directly or used within a graph call; within graphs the
            parameters have already been refreshed so the initialise graphs function fixes them in place.

        Returns
        -------
        Float
            Barrier option price.

        """
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_barriers(S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                                  barrier_direction=barrier_direction, knock=knock, 
                                  option=option)
        
        # Down and In Call
        if self.barrier_direction == 'down' and self.knock == 'in' and self.option == 'call':
            self.eta = 1
            self.phi = 1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.C + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.B + self.D + self.E
            

        # Up and In Call    
        if self.barrier_direction == 'up' and self.knock == 'in' and self.option == 'call':
            self.eta = -1
            self.phi = 1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.A + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.B - self.C + self.D + self.E


        # Down and In Put
        if self.barrier_direction == 'down' and self.knock == 'in' and self.option == 'put':
            self.eta = 1
            self.phi = -1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.B - self.C + self.D + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.A + self.E
                
                
        # Up and In Put         
        if self.barrier_direction == 'up' and self.knock == 'in' and self.option == 'put':
            self.eta = -1
            self.phi = -1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.B + self.D + self.E
            if self.K < self.H:
                self.opt_barrier_payoff = self.C + self.E
                
                
        # Down and Out Call
        if self.barrier_direction == 'down' and self.knock == 'out' and self.option == 'call':
            self.eta = 1
            self.phi = 1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.C + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.B - self.D + self.F
            
            
        # Up and Out Call
        if self.barrier_direction == 'up' and self.knock == 'out' and self.option == 'call':
            self.eta = -1
            self.phi = 1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.B + self.C - self.D + self.F


        # Down and Out Put
        if self.barrier_direction == 'down' and self.knock == 'out' and self.option == 'put':
            self.eta = 1
            self.phi = -1
            
            if self.K > self.H:
                self.opt_barrier_payoff = self.A - self.B + self.C - self.D + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.F
                
        # Up and Out Put         
        if self.barrier_direction == 'up' and self.knock == 'out' and self.option == 'put':
            self.eta = -1
            self.phi = -1
        
            if self.K > self.H:
                self.opt_barrier_payoff = self.B - self.D + self.F
            if self.K < self.H:
                self.opt_barrier_payoff = self.A - self.C + self.F

        return self.opt_barrier_payoff    


    def visualize(self, risk=None, S=None, T=None, r=None, q=None, sigma=None, option=None, 
                  direction=None, greek=None, graphtype=None, x_plot=None, y_plot=None, 
                  G1=None, G2=None, G3=None, T1=None, T2=None, T3=None, time_shift=None, 
                  interactive=None, notebook=None, colorscheme=None, colorintensity=None, 
                  size=None, axis=None, spacegrain=None, K=None, K1=None, K2=None, 
                  K3=None, K4=None, cash=None, ratio=None, value=None, combo_payoff=None):
        """
        Plot the chosen graph of risk or payoff.
        
        
        Parameters
        ----------
        risk : Bool
            Whether to display risk graph or payoff graph. The default is True.
        S : Float
            Underlying Stock Price. Used in risk & payoff graphs. The default is 100. 
        T : Float
            Time to Maturity. Used in risk & payoff graphs. The default is 0.25 (3 months).
        r : Float
            Interest Rate. Used in risk & payoff graphs. The default is 0.05 (5%).
        q : Float
            Dividend Yield. Used in risk & payoff graphs. The default is 0.
        sigma : Float
            Implied Volatility. Used in risk & payoff graphs. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. Used in risk & payoff graphs. The default is 'call'
        direction : Str
            Whether the payoff is long or short. Used in risk & payoff graphs. The default is 'long'.
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 'vega', 
            'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 'ultima', 
            'vega_bleed', 'charm'. Used in risk graphs. The default is 'delta'. 
        graphtype : Str
            Whether to plot 2D or 3D graph. Used in risk graphs. The default is 2D.
        x_plot : Str
            The x-axis variable ('price', 'strike', 'vol' or 'time'). Used in 2D-risk graphs. 
            The default is 'time'.
        y_plot : Str
            The y-axis variable ('value', 'delta', 'gamma', 'vega' or 'theta'). 
            Used in 2D-risk graphs. The default is 'delta.
        G1 : Float
            Strike Price of option 1. Used in 2D-risk graphs. The default is 90.
        G2 : Float
            Strike Price of option 2. Used in 2D-risk graphs. The default is 100.
        G3 : Float
            Strike Price of option 3. Used in 2D-risk graphs. The default is 110.
        T1 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The default is 0.25 (3 months).
        T2 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The default is 0.25 (3 months).
        T3 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The default is 0.25 (3 months).
        time_shift : Float
            Difference between T1 and T2 in rho graphs. Used in 2D-risk graphs. 
            The default is 0.25 (3 months).
        interactive : Bool
            Whether to show matplotlib (False) or plotly (True) graph. Used in 3D-risk graphs. 
            The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and hence whether 
            it should output in line or to an HTML file. Used in 3D-risk graphs. 
            The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. Used in 3D-risk graphs. 
            The default is 'jet' (which is a palette that works in both plotly and matplotlib).
        colorintensity : Float
            The alpha value indicating level of transparency / intensity. 
            The default is 1.
        size : Tuple
            Figure size for matplotlib chart. Used in 3D-risk graphs. The default is (12, 8).
        axis : Str
            Whether the x-axis is 'price' or 'vol'. Used in 3D-risk graphs. The default is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D graphs. Used in 3D-risk graphs. 
            The default is 100. 
        K : Float
            Strike Price of option 1. Used in payoff graphs. The default is 100 
            (individual payoffs may have own defaults).
        K1 : Float
             Strike Price of option 1. Used in payoff graphs. The default is 95 
             (individual payoffs may have own defaults).
        K2 : Float
             Strike Price of option 2. Used in payoff graphs. The default is 105 
             (individual payoffs may have own defaults).
        K3 : Float
             Strike Price of option 3. Used in payoff graphs. The default is 105 
             (individual payoffs may have own defaults).
        K4 : Float
             Strike Price of option 4. Used in payoff graphs. The default is 105 
             (individual payoffs may have own defaults).
        cash : Bool
            Whether to discount forward to present value. Used in forward payoff graph. The default is False.
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. Used in backspread, 
            ratio vertical spread payoff graphs. The default is 2. 
        value : Bool
            Whether to show the current value as well as the terminal payoff. Used in payoff graphs. 
            The default is False.
        combo_payoff : Str
            The payoff to be displayed. Used in payoff graphs. The default is 'straddle'. 
   
        Returns
        -------
        Displays graph of either 2D / 3D greeks or payoff diagram.

        """

        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(risk=risk, S=S, T=T, r=r, q=q, sigma=sigma, option=option, 
                              direction=direction, greek=greek, graphtype=graphtype, 
                              x_plot=x_plot, y_plot=y_plot,  G1=G1, G2=G2, G3=G3, 
                              T1=T1, T2=T2, T3=T3, time_shift=time_shift, interactive=interactive, 
                              notebook=notebook, colorscheme=colorscheme, colorintensity=colorintensity, 
                              size=size,  axis=axis, spacegrain=spacegrain, K=K, 
                              K1=K1, K2=K2, K3=K3, K4=K4, cash=cash, ratio=ratio, 
                              value=value, combo_payoff=combo_payoff)
        
        if self.risk == True:
            return self.greeks(x_plot=self.x_plot, y_plot=self.y_plot, S=self.S, 
                               G1=self.G1, G2=self.G2, G3=self.G3, T=self.T, T1=self.T1, 
                               T2=self.T2, T3=self.T3, time_shift=self.time_shift, 
                               r=self.r, q=self.q, sigma=self.sigma, option=self.option, 
                               direction=self.direction, interactive=self.interactive, 
                               notebook=self.notebook, colorscheme=self.colorscheme, 
                               colorintensity=self.colorintensity, size=self.size, 
                               axis=self.axis, spacegrain=self.spacegrain, greek=self.greek, 
                               graphtype=self.graphtype)
        
        if self.risk == False:
            return self.payoffs(S=S, K=K, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                                sigma=sigma, option=option, direction=direction, 
                                cash=cash, ratio=ratio, value=value, combo_payoff=combo_payoff)
        
    
    def greeks(self, x_plot=None, y_plot=None, S=None, G1=None, G2=None, 
               G3=None, T=None, T1=None, T2=None, T3=None, time_shift=None, r=None, 
               q=None, sigma=None, option=None, direction=None, interactive=None, 
               notebook=None, colorscheme=None, colorintensity=None, size=None, 
               axis=None, spacegrain=None, greek=None, graphtype=None):
        """
        Plot the chosen 2D or 3D graph
        
        
        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 'vega', 
            'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 'ultima', 
            'vega_bleed', 'charm'. The default is 'delta'
        x_plot : Str
                 The x-axis variable ('price', 'strike', 'vol' or 'time'). The 
                 default is 'time'.
        y_plot : Str
                 The y-axis variable ('value', 'delta', 'gamma', 'vega' or 'theta'). 
                 The default is 'delta.
        S : Float
             Underlying Stock Price. The default is 100.
        G1 : Float
             Strike Price of option 1. The default is 90.
        G2 : Float
             Strike Price of option 2. The default is 100.
        G3 : Float
             Strike Price of option 3. The default is 110.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        T1 : Float
             Time to Maturity of option 1. The default is 0.25 (3 months).
        T2 : Float
             Time to Maturity of option 1. The default is 0.25 (3 months).
        T3 : Float
             Time to Maturity of option 1. The default is 0.25 (3 months).
        time_shift : Float
             Difference between T1 and T2 in rho graphs. The default is 0.25 (3 months).
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
        interactive : Bool
            Whether to show matplotlib (False) or plotly (True) graph. The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and hence whether 
            it should output in line or to an HTML file. The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. The default is 'jet' 
            (which works in both).
        colorintensity : Float
            The alpha value indicating level of transparency / intensity. The default is 1.
        size : Tuple
            Figure size for matplotlib chart. The default is (12, 8).
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D graphs. The default is 100.    
        graphtype : Str
            Whether to plot 2D or 3D graph. The default is 2D.

        Returns
        -------
        Runs method to display either 2D or 3D greeks graph.

        """
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, G3=G3, 
                              T=T, T1=T1, T2=T2, T3=T3, time_shift=time_shift, r=r, 
                              q=q, sigma=sigma, option=option, interactive=interactive, 
                              notebook=notebook, colorscheme=colorscheme, colorintensity=colorintensity, 
                              size=size, direction=direction, axis=axis, spacegrain=spacegrain, 
                              greek=greek, graphtype=graphtype)
        
        # Run 2D greeks method
        if self.graphtype == '2D':
            self.greeks_graphs_2D(x_plot=self.x_plot, y_plot=self.y_plot, 
                                  S=self.S, G1=self.G1, G2=self.G2, G3=self.G3, 
                                  T=self.T, T1=self.T1, T2=self.T2, T3=self.T3, 
                                  time_shift=self.time_shift, r=self.r, q=self.q, 
                                  sigma=self.sigma, option=self.option, direction=self.direction)
        
        # Run 3D greeks method    
        if self.graphtype == '3D':
            self.greeks_graphs_3D(S=self.S, r=self.r, q=self.q, sigma=self.sigma, 
                                  option=self.option, interactive=self.interactive, 
                                  notebook=self.notebook, colorscheme=self.colorscheme, 
                                  colorintensity=self.colorintensity, size=self.size, 
                                  direction=self.direction, axis=self.axis, spacegrain=self.spacegrain, 
                                  greek=self.greek)
    
    
    def greeks_graphs_2D(self, x_plot=None, y_plot=None, S=None, G1=None, G2=None, 
                         G3=None, T=None, T1=None, T2=None, T3=None, time_shift=None, 
                         r=None, q=None, sigma=None, option=None, direction=None):
        """
        Plot chosen 2D greeks graph.
                

        Parameters
        ----------
        x_plot : Str
                 The x-axis variable ('price', 'strike', 'vol' or 'time'). The 
                 default is 'time'.
        y_plot : Str
                 The y-axis variable ('value', 'delta', 'gamma', 'vega' or 'theta'). 
                 The default is 'delta.
        S : Float
             Underlying Stock Price. The default is 100.
        G1 : Float
             Strike Price of option 1. The default is 90.
        G2 : Float
             Strike Price of option 2. The default is 100.
        G3 : Float
             Strike Price of option 3. The default is 110.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        T1 : Float
             Time to Maturity of option 1. The default is 0.25 (3 months).
        T2 : Float
             Time to Maturity of option 1. The default is 0.25 (3 months).
        T3 : Float
             Time to Maturity of option 1. The default is 0.25 (3 months).
        time_shift : Float
             Difference between T1 and T2 in rho graphs. The default is 0.25 (3 months).
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

        Returns
        -------
        Runs method to create data for 2D greeks graph.

        """
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, 
                              G3=G3, T=T, T1=T1, T2=T2, T3=T3, time_shift=time_shift, 
                              r=r, q=q, sigma=sigma, option=option, direction=direction)
        
        # create arrays of 1000 equally spaced points for a range of strike prices, 
        # volatilities and maturities
        self.SA = np.linspace(0.8 * self.S, 1.2 * self.S, 1000)
        self.sigmaA = np.linspace(0.05, 0.5, 1000)
        self.TA = np.linspace(0.01, 1, 1000)
        self.S0 = self.S
    
        self._2D_general_graph()       
    

    def _2D_general_graph(self):                               
        """
        Creates data for 2D greeks graph.

        Returns
        -------
        Runs method to graph using Matplotlib.

        """
        # y-axis parameters other than rho require 3 options to be graphed
        if self.y_plot in self.y_name_dict.keys():
            for opt in [1, 2, 3]:
                if self.x_plot == 'price':
                    
                    # Use self.__dict__ to access names, C1... etc., dynamically
                    # Use getattr() with the y-plot value in y_name_dict to access each method, 'delta'... etc., dynamically
                    # For price we set S to the array SA 
                    self.__dict__['C'+str(opt)] = getattr(self, self.y_name_dict[self.y_plot])(S=self.SA, K=self.__dict__['G'+str(opt)], T=self.__dict__['T'+str(opt)], 
                                                                             r=self.r, q=self.q, sigma=self.sigma, option=self.option, refresh='graph')
                if self.x_plot == 'vol':
                    
                    # For vol we set sigma to the array sigmaA
                    self.__dict__['C'+str(opt)] = getattr(self, self.y_name_dict[self.y_plot])(S=self.S0, K=self.__dict__['G'+str(opt)], T=self.__dict__['T'+str(opt)], 
                                                                              r=self.r, q=self.q, sigma=self.sigmaA, option=self.option, refresh='graph')
                if self.x_plot == 'time':
                    
                    # For time we set T to the array TA
                    self.__dict__['C'+str(opt)] = getattr(self, self.y_name_dict[self.y_plot])(S=self.S0, K=self.__dict__['G'+str(opt)], T=self.TA, r=self.r, 
                                                                                   q=self.q, sigma=self.sigma, option=self.option, refresh='graph')
            
            # Reverse the option value if direction is 'short'        
            if self.direction == 'short':
                for opt in [1, 2, 3]:
                    self.__dict__['C'+str(opt)] = -self.__dict__['C'+str(opt)]
            
            # Call strike_tenor_label method to assign labels to chosen strikes and tenors
            self._strike_tenor_label()
 
        # rho requires 4 options to be graphed 
        if self.y_plot == 'rho':
            
            # Set T1 and T2 to the specified time and shifted time
            self.T1 = self.T
            self.T2 = self.T + self.time_shift
            
            # 2 Tenors
            tenor_type = {1:1, 2:2, 3:1, 4:2}
            
            # And call and put for each tenor 
            opt_type = {1:'call', 2:'call', 3:'put', 4:'put'}
            for opt in [1, 2, 3, 4]:
                if self.x_plot == 'price':
                    
                    # For price we set S to the array SA
                    self.__dict__['C'+str(opt)] = getattr(self, str(self.y_plot))(S=self.SA, K=self.G2, T=self.__dict__['T'+str(tenor_type[opt])], 
                                                                             r=self.r, q=self.q, sigma=self.sigma, option=opt_type[opt], refresh='graph')
                if self.x_plot == 'strike':
                    
                    # For strike we set K to the array SA
                    self.__dict__['C'+str(opt)] = getattr(self, str(self.y_plot))(S=self.S0, K=self.SA, T=self.__dict__['T'+str(tenor_type[opt])],
                                                                             r=self.r, q=self.q, sigma=self.sigma, option=opt_type[opt], refresh='graph')
                if self.x_plot == 'vol':
                    
                    # For vol we set sigma to the array sigmaA
                    self.__dict__['C'+str(opt)] = getattr(self, str(self.y_plot))(S=self.S0, K=self.G2, T=self.__dict__['T'+str(tenor_type[opt])], 
                                                                             r=self.r, sigma=self.sigmaA, option=opt_type[opt], refresh='graph')
            
            # Reverse the option value if direction is 'short'        
            if self.direction == 'short':
                for opt in [1, 2, 3, 4]:
                    self.__dict__['C'+str(opt)] = -self.__dict__['C'+str(opt)]
    
            # Assign the option labels
            self.label1 = str(int(self.T1*365))+' Day Call'
            self.label2 = str(int(self.T2*365))+' Day Call'
            self.label3 = str(int(self.T1*365))+' Day Put'
            self.label4 = str(int(self.T2*365))+' Day Put'
    
        # Convert the x-plot and y-plot values to axis labels   
        self.xlabel = self.label_dict[str(self.x_plot)]
        self.ylabel = self.label_dict[str(self.y_plot)]
        
        # If the greek is rho or the same for a call or a put, set the option name to 'Call / Put' 
        if self.y_plot in [self.equal_greeks, 'rho']:
                self.option = 'Call / Put'     
        
        # Create chart title as direction plus option type plus y-plot vs x-plot    
        self.title = (str(self.direction.title())+' '+str(self.option.title())+
                      ' '+self.y_plot.title()+' vs '+self.x_plot.title())   
        
        # Set the x-axis array as price, vol or time
        self.x_name = str(self.x_plot)
        if self.x_name in self.x_name_dict.keys():
            self.xarray = (self.__dict__[str(self.x_name_dict[self.x_name])] * 
                           self.x_scale_dict[self.x_name])
        
        # Plot 3 option charts    
        if self.y_plot in self.y_name_dict.keys():        
            self._vis_greeks_mpl(yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                                 xarray=self.xarray, label1=self.label1, label2=self.label2, 
                                 label3=self.label3, xlabel=self.xlabel, ylabel=self.ylabel, 
                                 title=self.title)       
        # Plot Rho charts    
        elif self.y_plot == 'rho':
            self._vis_greeks_mpl(yarray1=self.C1, yarray2=self.C2, yarray3=self.C3, 
                                 yarray4=self.C4, xarray=self.xarray, label1=self.label1, 
                                 label2=self.label2, label3=self.label3, label4=self.label4, 
                                 xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
 
        else:
            print("Graph not printed")
    
    
    def _strike_tenor_label(self):
        """
        Assign labels to chosen strikes and tenors in 2D greeks graph

        Returns
        -------
        Str
            Labels for each of the 3 options in 2D greeks graph.

        """
        self.strike_label = dict()
        for key, value in {'G1':'label1', 'G2':'label2', 'G3':'label3'}.items():
            
            # If the strike is 100% change name to 'ATM'
            if self.__dict__[str(key)] == self.S0:
                self.strike_label[value] = 'ATM Strike'
            else:
                self.strike_label[value] = str(int(self.__dict__[key]))+' Strike' 
               
        for k, v in {'T1':'label1', 'T2':'label2', 'T3':'label3'}.items():
            
            # Make each label value the number of days to maturity plus the strike level
            self.__dict__[v] = str(int(self.__dict__[str(k)]*365))+' Day '+self.strike_label[str(v)]
                
        return self            


    def _vis_greeks_mpl(self, xarray=None, yarray1=None, yarray2=None, yarray3=None, 
                        yarray4=None, label1=None, label2=None, label3=None, label4=None, 
                        xlabel=None, ylabel=None, title='Payoff'):
        """
        Display the 2D greeks chart using matplotlib

        Parameters
        ----------
        xarray : Array
            x-axis values. 
        yarray1 : Array
            y-axis values for option 1. 
        yarray2 : Array
            y-axis values for option 2.
        yarray3 : Array
            y-axis values for option 3.
        yarray4 : Array
            y-axis values for option 4.
        label1 : Str
            Option 1 label.
        label2 : Str
            Option 2 label.
        label3 : Str
            Option 3 label.
        label4 : Str
            Option 4 label.
        xlabel : Str
            x-axis label.
        ylabel : Str
            y-axis label.
        title : Str
            Chart title.

        Returns
        -------
        2D Greeks chart.

        """
        
        # Set style to Seaborn Darkgrid
        plt.style.use('seaborn-darkgrid')
        
        # Create the figure and axes objects
        fig, ax = plt.subplots()
        
        # Plot the 1st option
        ax.plot(xarray, yarray1, color='blue', label=label1)
        
        # Plot the 2nd option
        ax.plot(xarray, yarray2, color='red', label=label2)
        
        # Plot the 3rd option
        ax.plot(xarray, yarray3, color='green', label=label3)
        
        # 4th option only used in Rho graphs
        if label4 is not None:
            ax.plot(xarray, yarray4, color='orange', label=label4)
        
        # Apply a grid
        plt.grid(True)
        
        # Set x and y axis labels and title
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
 
        # Create a legend 
        ax.legend()
        
        # Display the chart
        plt.show()
    
    
    def greeks_graphs_3D(self, S=None, r=None, q=None, sigma=None, 
                         option=None, interactive=None, notebook=None, colorscheme=None, 
                         colorintensity=None, size=None, direction=None, axis=None, 
                         spacegrain=None, greek=None):
        """
        
        Plot chosen 3D greeks graph.

        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 'vega', 
            'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 'ultima', 
            'vega_bleed', 'charm'. The default is 'delta'
        S : Float
             Underlying Stock Price. The default is 100.
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        option : Str
            Option type, Put or Call. The default is 'call'
        interactive : Bool
            Whether to show matplotlib (False) or plotly(True) graph. The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and hence whether 
            it should output in line or to an HTML file. The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. The default is 'jet' 
            (which works in both).
        colorintensity : Float
            The alpha value indicating level of transparency / intensity. The default is 1.
        size : Tuple
            Figure size for matplotlib chart. The default is (12, 8).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D graphs. The default is 100.        

        Returns
        -------
        Runs method to display 3D greeks graph.

        """
        
        # Pass parameters to be initialised. If not provided they will be populated with default values
        self._initialise_func(greek=greek, S=S, r=r, q=q, sigma=sigma, option=option, 
                              interactive=interactive, notebook=notebook, colorscheme=colorscheme, 
                              colorintensity=colorintensity, size=size, direction=direction, 
                              axis=axis, spacegrain=spacegrain)
        
        # Select the input name and method name from the greek dictionary 
        for greek_label, greek_func in self.greek_dict.items():
            
            # If the greek is the same for call or put, set the option value to 'Call / Put'
            if self.greek in self.equal_greeks:
                self.option = 'Call / Put'
            
            # For the specified greek
            if self.greek == greek_label:

                # Prepare the graph axes                 
                self._graph_space_prep()
 
                if self.axis == 'price':
                    
                    # Select the individual greek method using getattr()
                    self.z = getattr(self, greek_func)(S=self.x, K=self.S, T=self.y, 
                                                       r=self.r, sigma=self.sigma, 
                                                       option=self.option, refresh='graph')
                
                if self.axis == 'vol':
                    
                    # Select the individual greek method using getattr()
                    self.z = getattr(self, greek_func)(S=self.S, K=self.S, T=self.y, 
                                                       r=self.r, sigma=self.x, 
                                                       option=self.option, refresh='graph')
        
        # Run the 3D visualisation method            
        self._vis_greeks_3D()            
    
    
    def _graph_space_prep(self):
        """
        Prepare the axis ranges to be used in 3D graph.

        Parameters
        ----------
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default is 'price'.

        Returns
        -------
        Various
            Updated parameters to be used in 3D graph.

        """
        
        # Select the strike and Time ranges for each greek from the 3D chart ranges dictionary 
        self.SA_lower = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['SA_lower']
        self.SA_upper = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['SA_upper']
        self.TA_lower = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['TA_lower']
        self.TA_upper = self.df_dict['df_3D_chart_ranges'][str(self.greek)]['TA_upper']
        
        # Set the volatility range from 5% to 50%
        self.sigmaA_lower = 0.05 
        self.sigmaA_upper = 0.5 

        # create arrays of 100 equally spaced points for the ranges of strike prices, 
        # volatilities and maturities
        self.SA = np.linspace(self.SA_lower * self.S, self.SA_upper * self.S, int(self.spacegrain))
        self.TA = np.linspace(self.TA_lower, self.TA_upper, int(self.spacegrain))
        self.sigmaA = np.linspace(self.sigmaA_lower, self.sigmaA_upper, int(self.spacegrain))
        
        # set y-min and y-max labels 
        self.ymin = self.TA_lower
        self.ymax = self.TA_upper
        self.axis_label2 = 'Time to Expiration (Days)'
        
        # set x-min and x-max labels 
        if self.axis == 'price':
            self.x, self.y = np.meshgrid(self.SA, self.TA)
            self.xmin = self.SA_lower
            self.xmax = self.SA_upper
            self.graph_scale = 1
            self.axis_label1 = 'Underlying Value'            
            
        if self.axis == 'vol':
            self.x, self.y = np.meshgrid(self.sigmaA, self.TA)
            self.xmin = self.sigmaA_lower
            self.xmax = self.sigmaA_upper    
            self.graph_scale = 100
            self.axis_label1 = 'Volatility %'    

        return self
    
   
    def _vis_greeks_3D(self):
        """
        Display 3D greeks graph.

        Returns
        -------
        If 'interactive' is False, a matplotlib static graph.
        If 'interactive' is True, a plotly graph that can be rotated and zoomed.

        """
        
        # Reverse the z-axis data if direction is 'short'
        if self.direction == 'short':
            self.z = -self.z
        
        # Label the graph based on whether it is different for calls & puts or the same
        if self.option == 'Call / Put':
            titlename = str(str(self.direction.title())+' '+self.option+' Option '+str(self.greek.title()))
        else:    
            titlename = str(str(self.direction.title())+' '+str(self.option.title())+
                            ' Option '+str(self.greek.title()))
           
        # Create a matplotlib graph    
        if self.interactive == False:
        
            # create figure with specified size tuple
            fig = plt.figure(figsize=self.size)
            ax = fig.add_subplot(111, projection='3d')
            
            # apply graph_scale so that if volatility is the x-axis it wil be * 100
            ax.plot_surface(self.x * self.graph_scale,
                            self.y * 365,
                            self.z,
                            rstride=2, cstride=2,
                            
                            # set the colormap to the chosen colorscheme
                            cmap=plt.get_cmap(self.colorscheme),
                            
                            # set the alpha value to the chosen colorintensity
                            alpha=self.colorintensity,
                            linewidth=0.25)
            
            # Auto scale the z-axis
            ax.set_zlim(auto=True)
            
            # Set fontsize of axis ticks
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Set x-axis to decrease from left to right
            ax.invert_xaxis()
            
            # Label axes
            ax.set_xlabel(self.axis_label1, fontsize=14, labelpad=15)
            ax.set_ylabel(self.axis_label2, fontsize=14, labelpad=15)
            ax.set_zlabel(str(self.greek.title()), fontsize=14, labelpad=15)
            
            # Specify title
            ax.set_title(titlename, fontsize=18)
            
            # Display graph
            plt.show()

        # Create a plotly graph
        if self.interactive == True:
            
            # Set the ranges for the contour values
            contour_x_start = self.ymin
            contour_x_stop = self.ymax * 360
            contour_x_size = contour_x_stop / 18
            contour_y_start = self.xmin
            contour_y_stop = self.xmax * self.graph_scale
            contour_y_size = int((self.xmax - self.xmin) / 20)
            contour_z_start = np.min(self.z)
            contour_z_stop = np.max(self.z)
            contour_z_size = int((np.max(self.z) - np.min(self.z)) / 10)
            
            
            # create plotly figure object
            fig = go.Figure(data=[go.Surface(x=self.y*365, 
                                             y=self.x*self.graph_scale, 
                                             z=self.z, 

                                             # set the colorscale to the chosen colorscheme
                                             colorscale=self.colorscheme, 
                                             
                                             # Define the contours
                                             contours = {"x": {"show": True, "start": contour_x_start, 
                                                               "end": contour_x_stop, "size": contour_x_size, "color":"white"},            
                                                         "y": {"show": True, "start": contour_y_start, 
                                                               "end": contour_y_stop, "size": contour_y_size, "color":"white"},  
                                                         "z": {"show": True, "start": contour_z_start, 
                                                               "end": contour_z_stop, "size": contour_z_size}},)])
            
            # Set initial view position
            camera = dict(
                eye=dict(x=2, y=1, z=1)
            )
            
            # Set x-axis to decrease from left to right
            fig.update_scenes(xaxis_autorange="reversed")
            # Set y-axis to increase from left to right
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
                                # Label axes
                                xaxis_title=self.axis_label2,
                                yaxis_title=self.axis_label1,
                                zaxis_title=str(self.greek.title()),),
                              title={'text':titlename,
                                     'y':0.9,
                                     'x':0.5,
                                     'xanchor':'center',
                                     'yanchor':'top',
                                     'font':dict(#family="Courier New, monospace",
                                               size=20,
                                               color="black")},
                              autosize=False, 
                              width=800, height=800,
                              margin=dict(l=65, r=50, b=65, t=90),
                              scene_camera=camera)
            
            # If running in an iPython notebook the chart will display in line
            if self.notebook == True:
                fig.show()
            
            # Otherwise create an HTML file that opens in a new window
            else:
                plot(fig, auto_open=True)
   
    
    def payoffs(self, S=None, K=None, K1=None, K2=None, K3=None, K4=None, 
                T=None, r=None, q=None, sigma=None, option=None, direction=None, 
                cash=None, ratio=None, value=None, combo_payoff=None):
        """
        Displays the graph of the specified combo payoff.
                
        Parameters
        ----------
        S : Float
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
            Time to Maturity. The default is 0.25 (3 months).
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
            self.call(S=S, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                      value=value)
        
        if combo_payoff == 'put':
            self.put(S=S, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                      value=value)
        
        if combo_payoff == 'stock':
            self.stock(S=S, direction=direction)
        
        if combo_payoff == 'forward':
            self.forward(S=S, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction,
                         cash=cash)
        
        if combo_payoff == 'collar':
            self.collar(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                        value=value)
        
        if combo_payoff == 'spread':
            self.spread(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                        direction=direction, value=value)
            
        if combo_payoff == 'backspread':
            self.backspread(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option, 
                            ratio=ratio, value=value)
        
        if combo_payoff == 'ratio vertical spread':
            self.ratio_vertical_spread(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                                       option=option, ratio=ratio, value=value)
        
        if combo_payoff == 'straddle':
            self.straddle(S=S, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                          value=value)

        if combo_payoff == 'strangle':
            self.strangle(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                          value=value)
        
        if combo_payoff == 'butterfly':    
            self.butterfly(S=S, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                           option=option, direction=direction, value=value)
        
        if combo_payoff == 'christmas tree':
            self.christmas_tree(S=S, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                                option=option, direction=direction, value=value)    

        if combo_payoff == 'condor':
            self.condor(S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                        sigma=sigma, option=option, direction=direction, value=value)
        
        if combo_payoff == 'iron butterfly':
            self.iron_butterfly(S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                                sigma=sigma, direction=direction, value=value)
            
        if combo_payoff == 'iron condor':
            self.iron_condor(S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                             sigma=sigma, direction=direction, value=value)
            
    
    def call(self, S=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, 
             value=None):
        """
        Displays the graph of the call.

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of option 1. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                              value=value, option1=self.option1)
        
        # Calculate option prices
        self._return_options(legs=1)
        
        # Create payoff based on direction
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
        
        # Visualize payoff        
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')   
                
        
    def put(self, S=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, 
            value=None):
        """
        Displays the graph of the put.

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of option 1. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                              value=value, option1=self.option1)
        
        # Calculate option prices
        self._return_options(legs=1)
        
        # Create payoff based on direction
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
        
        # Visualize payoff        
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')   
               
        
    def stock(self, S=None, direction=None):
        """
        Displays the graph of the underlying.

        Parameters
        ----------
        S : Float
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
        self._initialise_func(S=S, direction=direction)
        
        # Define strike range
        self.SA = np.linspace(0.75 * self.S, 1.25 * self.S, 1000)
        
        # Create payoff based on option type
        if self.direction == 'long':
            payoff = self.SA - self.S
            title = 'Long Stock'
        
        if self.direction == 'short':
            payoff = self.S - self.SA
            title = 'Short Stock'
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, label='Payoff', 
                         title=title)     
            
    
    def forward(self, S=None, T=None, r=None, q=None, sigma=None, direction=None, 
                cash=None):
        """
        Displays the graph of the synthetic forward strategy:
            Long one ATM call
            Short one ATM put

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, T=T, r=r, q=q, sigma=sigma, option1='call', 
                              option2='put', direction=direction, cash=cash)
        
        # Set both strikes to spot and both maturities to input maturity 
        self.K1 = self.S
        self.K2 = self.S
        self.T1 = self.T
        self.T2 = self.T
        
        # Calculate option prices
        self._return_options(legs=2)
        
        # Whether to discount the payoff
        if self.cash == False:
            pv = 1
        else:
            pv = self.discount
        
        # Create payoff based on option type
        if self.direction == 'long':
            payoff = (self.C1 - self.C2 - self.C1_0 + self.C2_0) * pv
            title = 'Long Forward'
            
        if self.direction == 'short':
            payoff = -self.C1 + self.C2 + self.C1_0 - self.C2_0 * pv
            title = 'Short Forward'
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, label='Payoff', 
                         title=title)
    
    
    def collar(self, S=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
               direction=None, value=None):
        """
        Displays the graph of the collar strategy:
            Long one OTM put
            Short one OTM call

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 98.
        K2 : Float
             Strike Price of option 2. The default is 102.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma,
                              option1='put', option2='call', direction=direction, 
                              value=value)

        # Calculate option prices
        self._return_options(legs=2)
        
        # Create payoff based on option type
        if self.direction == 'long':
            payoff = self.SA - self.S + self.C1 - self.C2 - self.C1_0 + self.C2_0
            title = 'Long Collar'
            if self.value == True:
                payoff2 = self.SA - self.S + self.C1_G - self.C2_G - self.C1_0 + self.C2_0
            else:
                payoff2 = None
                
        if self.direction == 'short':
            payoff = -self.SA + self.S - self.C1 + self.C2 + self.C1_0 - self.C2_0
            title = 'Short Collar'
            if self.value == True:
                payoff2 = -self.SA + self.S - self.C1_G + self.C2_G + self.C1_0 - self.C2_0
            else:
                payoff2 = None
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')

    
    
    def spread(self, S=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
               option=None, direction=None, value=None):
        """
        Displays the graph of the spread strategy:
            Long one ITM option
            Short one OTM option

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option=option, option1=option, option2=option, direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=2)
 
        # Create payoff based on option type
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
        
        # Create title based on option type and direction       
        if self.option == 'call' and self.direction == 'long':
            title = 'Bull Call Spread'
        if self.option == 'put' and self.direction == 'long':
            title = 'Bull Put Spread'
        if self.option == 'call' and self.direction == 'short':
            title = 'Bear Call Spread'
        if self.option == 'put' and self.direction == 'short':
            title = 'Bear Put Spread' 
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
   
    def backspread(self, S=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
                   option=None, ratio=None, value=None):
        """
        Displays the graph of the backspread strategy:
            Short one ITM option
            Long ratio * OTM options

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option=option, option1=option, option2=option, ratio=ratio, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=2)
        
        # Create payoff based on option type
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
        
        # Visualize payoff        
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
        
    def ratio_vertical_spread(self, S=None, K1=None, K2=None, T=None, r=None, q=None, 
                              sigma=None, option=None, ratio=None, value=None):
        """
        Displays the graph of the ratio vertical spread strategy:
            Long one ITM option
            Short ratio * OTM options

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option=option, option1=option, option2=option, ratio=ratio, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=2)
        
        # Create payoff based on option type
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
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
        
    
    def straddle(self, S=None, K=None, T=None, r=None, q=None, sigma=None, direction=None, 
                 value=None):
        """
        Displays the graph of the straddle strategy:
            Long one ATM put
            Long one ATM call

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of options 1 and 2. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K=K, K1=K, K2=K, T=T, T1=T, T2=T, r=r, q=q, 
                              sigma=sigma, option1='put', option2='call', direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=2)
        
        # Create payoff based on direction
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
        
        # Visualize payoff    
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, 
                         payoff2=payoff2, label='Payoff', label2='Value')    
  
    
    def strangle(self, S=None, K1=None, K2=None, T=None, r=None, q=None, sigma=None, 
                 direction=None, value=None):
        """
        Displays the graph of the strangle strategy:
            Long one OTM put
            Long one OTM call

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, T=T, T1=T, T2=T, r=r, q=q, sigma=sigma, 
                              option1='put', option2='call', direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=2)
        
        # Create payoff based on direction
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
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')    


    def butterfly(self, S=None, K1=None, K2=None, K3=None, T=None, r=None, q=None, 
                  sigma=None, option=None, direction=None, value=None):
        """
        Displays the graph of the butterfly strategy:
            Long one ITM option
            Short two ATM options
            Long one OTM option

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 105.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, K3=K3, T=T, T1=T, T2=T, 
                              T3=T, r=r, q=q, sigma=sigma, option=option, option1=option, 
                              option2=option, option3=option, direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=3)
        
        # Create payoff based on direction
        if self.direction == 'long':
            payoff = (self.C1 - 2*self.C2 + self.C3 - self.C1_0 + 2*self.C2_0 - self.C3_0)
            if self.value == True:
                payoff2 = (self.C1_G - 2*self.C2_G + self.C3_G - self.C1_0 + 2*self.C2_0 - self.C3_0)
            else:
                payoff2 = None
                
        if self.direction == 'short':    
            payoff = (-self.C1 + 2*self.C2 - self.C3 + self.C1_0 - 2*self.C2_0 + self.C3_0)
            if self.value == True:
                payoff2 = (-self.C1_G + 2*self.C2_G - self.C3_G + self.C1_0 - 2*self.C2_0 + self.C3_0)
            else:
                payoff2 = None
        
        # Create title based on option type and direction                 
        if self.option == 'call' and self.direction == 'long':
            title = 'Long Butterfly with Calls'
        if self.option == 'put' and self.direction == 'long':
            title = 'Long Butterfly with Puts'
        if self.option == 'call' and self.direction == 'short':
            title = 'Short Butterfly with Calls'
        if self.option == 'put' and self.direction == 'short':
            title = 'Short Butterfly with Puts'
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')

    
    def christmas_tree(self, S=None, K1=None, K2=None, K3=None, T=None, r=None, 
                       q=None, sigma=None, option=None, direction=None, value=None):
        """
        Displays the graph of the christmas tree strategy:
            Long one ITM option
            Short one ATM option
            Short one OTM option

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 105.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, K3=K3, T=T, T1=T, T2=T, 
                              T3=T, r=r, q=q, sigma=sigma, option=option, option1=option, 
                              option2=option, option3=option, direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=3)
        
        # Create payoff based on option type and direction
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
        
        # Visualize payoff    
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')


    def condor(self, S=None, K1=None, K2=None, K3=None, K4=None, T=None, r=None, 
               q=None, sigma=None, option=None, direction=None, value=None):
        """
        Displays the graph of the condor strategy:
            Long one low strike option
            Short one option with a higher strike
            Short one option with a higher strike 
            Long one option with a higher strike        

        Parameters
        ----------
        S : Float
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
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, T1=T, T2=T, 
                              T3=T, T4=T, r=r, q=q, sigma=sigma, option=option, option1=option, 
                              option2=option, option3=option, option4=option, direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=4)
        
        # Create payoff based on direction
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
        
        # Create title based on option type and direction        
        if self.option == 'call' and self.direction == 'long':
            title = 'Long Condor with Calls'
        if self.option == 'put' and self.direction == 'long':
            title = 'Long Condor with Puts'
        if self.option == 'call' and self.direction == 'short':
            title = 'Short Condor with Calls'
        if self.option == 'put' and self.direction == 'short':
            title = 'Short Condor with Puts'    
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')


    def iron_butterfly(self, S=None, K1=None, K2=None, K3=None, K4=None, T=None, r=None, 
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
        S : Float
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
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, T1=T, T2=T, 
                              T3=T, T4=T, r=r, q=q, sigma=sigma, option1='put', 
                              option2='call', option3='put', option4='call', direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=4)
        
        # Create payoff based on direction
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
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')
    
    
    def iron_condor(self, S=None, K1=None, K2=None, K3=None, K4=None, T=None, r=None, 
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
        S : Float
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
            Time to Maturity. The default is 0.25 (3 months).
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
        self._initialise_func(S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, T1=T, T2=T, 
                              T3=T, T4=T, r=r, q=q, sigma=sigma, option1='put', 
                              option2='put', option3='call', option4='call', direction=direction, 
                              value=value)
        
        # Calculate option prices
        self._return_options(legs=4)
        
        # Create payoff based on direction and value flag
        if self.direction == 'long':
            payoff = (self.C1 - self.C2 - self.C3 + self.C4 - self.C1_0 + 
                      self.C2_0 + self.C3_0 - self.C4_0)
            if self.value == True:
                payoff2 = (self.C1_G - self.C2_G - self.C3_G + self.C4_G - self.C1_0 + 
                           self.C2_0 + self.C3_0 - self.C4_0)
            else:
                payoff2 = None
        
        elif self.direction == 'short':
            payoff = (-self.C1 + self.C2 + self.C3 - self.C4 + self.C1_0 - 
                      self.C2_0 - self.C3_0 + self.C4_0)
            if self.value == True:
                payoff2 = (-self.C1_G + self.C2_G + self.C3_G - self.C4_G + self.C1_0 - 
                           self.C2_0 - self.C3_0 + self.C4_0)
            else:
                payoff2 = None
              
        # Create graph title based on direction 
        if self.direction == 'long':
            title = 'Long Iron Condor'
        
        if self.direction == 'short':
            title = 'Short Iron Condor'
        
        # Visualize payoff
        self._vis_payoff(S=self.S, SA=self.SA, payoff=payoff, title=title, payoff2=payoff2, 
                         label='Payoff', label2='Value')

    
    def _vis_payoff(self, S=None, SA=None, payoff=None, label=None, title='Option Payoff', 
                    payoff2=None, label2=None, xlabel='Stock Price', ylabel='P&L'):
        """
        Display the payoff diagrams using matplotlib

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        SA : Array
             Range of Strikes to provide x-axis values. The default is 75% to 125%.
        payoff : Array
            Terminal payoff value less initial cost.
        label : Str
            Label for terminal payoff. The default is 'Payoff'
        title : Str
            Chart title giving name of combo. The default is 'Option Payoff'
        payoff2 : Array
            Current payoff value less initial cost.
        label2 : Str
            Label for current payoff value.
        xlabel : Str
            The x-axis label. The default is 'Stock Price'.
        ylabel : Str
            The y-axis label. The default is 'P&L'.

        Returns
        -------
        2D Payoff Graph.

        """
        
        # Use seaborn darkgrid style 
        plt.style.use('seaborn-darkgrid')
        
        # Create the figure and axes objects
        fig, ax = plt.subplots()
        
        # Plot the terminal payoff
        ax.plot(SA, payoff, color='blue', label=label)
        
        # If the value flag is selected, plot the payoff with the current time to maturity
        if payoff2 is not None:
            ax.plot(SA, payoff2, color='red', label=label2)
        
        # Set a horizontal line at zero P&L 
        ax.axhline(y=0, linewidth=0.5, color='k')
        
        #Set a vertical line at ATM strike
        ax.axvline(x=S, linewidth=0.5, color='k')
        
        # Apply a black border to the chart
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('1')          
        
        # Apply a grid
        plt.grid(True)
        
        # Set x and y axis labels and title
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        
        # Create a legend
        ax.legend()
        
        # Display the chart
        plt.show()
    
    
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
        
        # Store initial spot value
        self.S0 = self.S
        
        # create array of 1000 equally spaced points between 75% of initial underlying price and 125%
        self.SA = np.linspace(0.75 * self.S, 1.25 * self.S, 1000)
        
        # Calculate the current price of option 1       
        self.C1_0 = self.price(S=self.S0, K=self.K1, T=self.T1, r=self.r, q=self.q, 
                               sigma=self.sigma, option=self.option1, refresh='graph')
        
        # Calculate the prices at maturity for the range of strikes in SA of option 1
        self.C1 = self.price(S=self.SA, K=self.K1, T=0, r=self.r, q=self.q, sigma=self.sigma, 
                             option=self.option1, refresh='graph')
        
        # Calculate the current prices for the range of strikes in SA of option 1
        self.C1_G = self.price(S=self.SA, K=self.K1, T=self.T1, r=self.r, q=self.q, 
                               sigma=self.sigma, option=self.option1, refresh='graph')
        
        if legs > 1:
            # Calculate the current price of option 2
            self.C2_0 = self.price(S=self.S0, K=self.K2, T=self.T2, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option2, refresh='graph')
            
            # Calculate the prices at maturity for the range of strikes in SA of option 2
            self.C2 = self.price(S=self.SA, K=self.K2, T=0, r=self.r, q=self.q, sigma=self.sigma, 
                                 option=self.option2, refresh='graph')
            
            # Calculate the current prices for the range of strikes in SA of option 2
            self.C2_G = self.price(S=self.SA, K=self.K2, T=self.T2, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option2, refresh='graph')

        if legs > 2:
            # Calculate the current price of option 3
            self.C3_0 = self.price(S=self.S0, K=self.K3, T=self.T3, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option3, refresh='graph')
            
            # Calculate the prices at maturity for the range of strikes in SA of option 3
            self.C3 = self.price(S=self.SA, K=self.K3, T=0, r=self.r, q=self.q, 
                                 sigma=self.sigma, option=self.option3, refresh='graph')
            
            # Calculate the current prices for the range of strikes in SA of option 3
            self.C3_G = self.price(S=self.SA, K=self.K3, T=self.T3, r=self.r, q=self.q, sigma=self.sigma, 
                                   option=self.option3, refresh='graph')
        
        if legs > 3:
            # Calculate the current price of option 4
            self.C4_0 = self.price(S=self.S0, K=self.K4, T=self.T4, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option4, refresh='graph')
            
            # Calculate the prices at maturity for the range of strikes in SA of option 4
            self.C4 = self.price(S=self.SA, K=self.K4, T=0, r=self.r, q=self.q, sigma=self.sigma, 
                                 option=self.option4, refresh='graph')
            
            # Calculate the current prices for the range of strikes in SA of option 4
            self.C4_G = self.price(S=self.SA, K=self.K4, T=self.T4, r=self.r, q=self.q, 
                                   sigma=self.sigma, option=self.option4, refresh='graph')
        
        # Restore Spot value
        self.S = self.S0
        
        return self
        
    
