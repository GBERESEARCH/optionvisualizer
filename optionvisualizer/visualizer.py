import os
import glob
import math
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import optionvisualizer.visualizer_params as vp
import numpy as np
import plotly.graph_objects as go
import scipy.stats as si
from mpl_toolkits.mplot3d.axes3d import Axes3D
from operator import itemgetter
from plotly.offline import plot
from PIL import Image


class Option():
    
    def __init__(self):

        # Dictionary of parameter defaults
        self.df_dict = vp.vis_params_dict

        # Initialize fixed default parameters
        self._init_fixed_params()
        
        
    def _init_fixed_params(self):
        """
        Initialize fixed default parameters using values from parameters dict

        Returns
        -------
        Various parameters and dictionaries to the object.

        """
        # Parameters to overwrite mpl_style defaults
        self.mpl_params = self.df_dict['df_mpl_params']
        self.mpl_3d_params = self.df_dict['df_mpl_3d_params']
        
        # Greek names as function input and individual function names
        self.greek_dict = self.df_dict['df_greek_dict']
        
        # Greks where the values are the same for a call or a put
        self.equal_greeks = self.df_dict['df_equal_greeks']        

        # Payoffs requiring changes to default parameters
        self.mod_payoffs = self.df_dict['df_mod_payoffs']
        
        # Those parameters that need changing
        self.mod_params = self.df_dict['df_mod_params']
        
        # Combo parameter values differing from standard defaults
        self.combo_dict = self.df_dict['df_combo_dict']

        # Dictionary mapping function parameters to x axis labels for 2D graphs        
        self.x_name_dict = self.df_dict['df_x_name_dict']
        
        # Dictionary mapping scaling parameters to x axis labels for 2D graphs
        self.x_scale_dict = self.df_dict['df_x_scale_dict']
        
        # Dictionary mapping function parameters to y axis labels for 2D graphs
        self.y_name_dict = self.df_dict['df_y_name_dict']
        
        # Dictionary mapping function parameters to axis labels  for 3D graphs
        self.label_dict = self.df_dict['df_label_dict']
        
    
    def _refresh_params_default(self, **kwargs):
        """
        Set parameters for use in various pricing functions to the
        default values.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if 
            no data provided

        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object 
                # and to input dictionary
                v = self.df_dict['df_'+str(k)]
                self.__dict__[k] = v
                kwargs[k] = v 
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return kwargs        
         
  
    def _refresh_dist(self, S, K, T, r, q, sigma):
        """
        Calculate various parameters and distributions

        Returns
        -------
        Various
            Assigns parameters to the object

        """
        
        # Cost of carry as risk free rate less dividend yield
        b = r - q
        
        carry = np.exp((b - r) * T)
        discount = np.exp(-r * T)
                
        with np.errstate(divide='ignore'):
            d1 = ((np.log(S / K) + (b + (0.5 * sigma ** 2)) * T) 
                / (sigma * np.sqrt(T)))
            
            d2 = ((np.log(S / K) + (b - (0.5 * sigma ** 2)) * T) 
                / (sigma * np.sqrt(T)))
            
            # standardised normal density function
            nd1 = ((1 / np.sqrt(2 * np.pi)) * (np.exp(-d1 ** 2 * 0.5)))
            
            # Cumulative normal distribution function
            Nd1 = si.norm.cdf(d1, 0.0, 1.0)
            minusNd1 = si.norm.cdf(-d1, 0.0, 1.0)
            Nd2 = si.norm.cdf(d2, 0.0, 1.0)
            minusNd2 = si.norm.cdf(-d2, 0.0, 1.0)
        
        return b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, minusNd2

 
    def _refresh_combo_params_default(self, **kwargs):
        """
        Set parameters for use in various pricing functions

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults 
            if no data provided

        """
 
        # Certain combo payoffs (found in the mod_payoffs list) require 
        # specific default parameters
        if self.combo_payoff in self.mod_payoffs:
            for k, v in kwargs.items():
                if v is None:
                    
                    # These parameters are in the mod_params list
                    if k in self.mod_params:
                        try:
                            
                            # Extract these from the df_combo_dict
                            v = self.df_combo_dict[str(
                                self.combo_payoff)][str(k)]
                            
                            # Assign to input dictionary
                            kwargs[k] = v
                            
                        except:
                            
                            # Otherwise set to the standard default 
                            # value
                            v = self.df_dict['df_'+str(k)]
                            kwargs[k] = v
                    
                    if k not in self.mod_params:
                        v = self.df_dict['df_'+str(k)]
                    
                    # Now assign this to the object and input dictionary
                    self.__dict__[k] = v
                    kwargs[k] = v
                
                # If the parameter has been provided as an input, 
                # assign this to the object
                else:
                    self.__dict__[k] = v
           
        else:
            
            # For all the other combo_payoffs
            for k, v in kwargs.items():
                
                # If a parameter has not been provided
                if v is None:
                    
                    # Set it to the object value and assign to the object 
                    # and to input dictionary
                    v = self.df_dict['df_'+str(k)]
                    self.__dict__[k] = v
                    kwargs[k] = v
                
                # If the parameter has been provided as an input, 
                # assign this to the object
                else:
                    self.__dict__[k] = v
                        
        return kwargs            
   
    
    def _barrier_factors(self, S, K, H, R, T, r, q, sigma, phi, eta):
        """
        Calculate the barrier option specific parameters

        Returns
        -------
        Various
            Assigns parameters to the object

        """
        
        # Cost of carry as risk free rate less dividend yield
        b = r - q
        
        mu = (b - ((sigma ** 2) / 2)) / (sigma ** 2)
        
        lamb_da = (
            np.sqrt(mu ** 2 + ((2 * r) / sigma ** 2)))
        
        z = (
            (np.log(H / S) / (sigma * np.sqrt(T))) 
            + (lamb_da * sigma * np.sqrt(T)))
        
        x1 = (
            np.log(S / K) / (sigma * np.sqrt(T)) 
            + ((1 + mu) * sigma * np.sqrt(T)))
        
        x2 = (
            np.log(S / H) / (sigma * np.sqrt(T)) 
            + ((1 + mu) * sigma * np.sqrt(T)))
        
        y1 = (
            np.log((H ** 2) / (S * K)) 
            / (sigma * np.sqrt(T)) 
            + ((1 + mu) * sigma * np.sqrt(T)))
        
        y2 = (
            np.log(H / S) / (sigma * np.sqrt(T)) 
            + ((1 + mu) * sigma * np.sqrt(T)))
        
        carry = np.exp((b - r) * T)
        
        A = (
            (phi * S * carry 
             * si.norm.cdf((phi * x1), 0.0, 1.0)) 
            - (phi * K * np.exp(-r * T) 
               * si.norm.cdf(((phi * x1) 
                              - (phi * sigma 
                                 * np.sqrt(T))), 0.0, 1.0)))
            

        B = (
            (phi * S * carry 
             * si.norm.cdf((phi * x2), 0.0, 1.0)) 
            - (phi * K * np.exp(-r * T) 
               * si.norm.cdf(((phi * x2) 
                              - (phi * sigma 
                                 * np.sqrt(T))), 0.0, 1.0)))
        
        C = (
            (phi * S * carry 
             * ((H / S) ** (2 * (mu + 1))) 
             * si.norm.cdf((eta * y1), 0.0, 1.0)) 
            - (phi * K * np.exp(-r * T) 
               * ((H / S) ** (2 * mu)) 
               * si.norm.cdf(((eta * y1) 
                              - (eta * sigma 
                                 * np.sqrt(T))), 0.0, 1.0)))
        
        D = (
            (phi * S * carry 
             * ((H / S) ** (2 * (mu + 1))) 
             * si.norm.cdf((eta * y2), 0.0, 1.0)) 
            - (phi * K * np.exp(-r * T) 
               * ((H / S) ** (2 * mu)) 
               * si.norm.cdf(((eta * y2) 
                              - (eta * sigma 
                                 * np.sqrt(T))), 0.0, 1.0)))
    
        E = (
            (R * np.exp(-r * T)) 
            * (si.norm.cdf(
                ((eta * x2) 
                 - (eta * sigma * np.sqrt(T))), 0.0, 1.0) 
                - (((H / S) ** (2 * mu)) 
                   * si.norm.cdf(
                       ((eta * y2) 
                        - (eta * sigma 
                           * np.sqrt(T))), 0.0, 1.0))))
        
        F = (
            R * (((H / S) ** (mu + lamb_da)) 
                      * (si.norm.cdf((eta * z), 0.0, 1.0)) 
                      + (((H / S) ** (mu - lamb_da)) 
                         * si.norm.cdf(
                             ((eta * z) 
                              - (2 * eta * lamb_da * 
                                 sigma * np.sqrt(T))), 0.0, 1.0))))

        return {'A':A, 
                'B':B, 
                'C':C, 
                'D':D, 
                'E':E,
                'F':F}


    def price(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated. 

        Returns
        -------
        Float
            Black Scholes Option Price.

        """
                
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)    

        if option == "call":
            opt_price = ((S * carry * Nd1) 
                - (K * np.exp(-r * T) * Nd2))  
        if option == "put":
            opt_price = ((K * np.exp(-r * T) * minusNd2) 
                - (S * carry * minusNd1))
        
        np.nan_to_num(opt_price, copy=False)
                
        return opt_price


    def delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Delta. 

        """    
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
                                
        if option == 'call':
            opt_delta = carry * Nd1
        if option == 'put':
            opt_delta = carry * (Nd1 - 1)
            
        return opt_delta
    
    
    def theta(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated. 

        Returns
        -------
        Float
            Option Theta.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
                   
        if option == 'call':
            opt_theta = (
                ((-S * carry * nd1 * sigma ) / (2 * np.sqrt(T)) 
                - (b - r) * (S * carry * Nd1) 
                - (r * K) * np.exp(-r * T) * Nd2)
                / 100) 
        if option == 'put':   
            opt_theta = (
                ((-S * carry * nd1 * sigma ) / (2 * np.sqrt(T)) 
                + (b - r) * (S * carry * minusNd1) 
                + (r * K) * np.exp(-r * T) * minusNd2) 
                / 100)

        return opt_theta
    
    
    def gamma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated. 

        Returns
        -------
        Float
            Option Gamma.

        """
               
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_gamma = ((nd1 * carry) / (S * sigma * np.sqrt(T)))
        
        return opt_gamma
    
    
    def vega(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
             option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated. 

        Returns
        -------
        Float
            Option Vega.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)

        opt_vega = ((S * carry * nd1 * np.sqrt(T)) / 100)
        
        return opt_vega
    
    
    def rho(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Rho.

        """        
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        if option == 'call':
            opt_rho = (
                (T * K * np.exp(-r * T) * Nd2) 
                / 10000)
        if option == 'put':
            opt_rho = (
                (-T * K * np.exp(-r * T) * minusNd2) 
                / 10000)
            
        return opt_rho


    def vanna(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated. 

        Returns
        -------
        Float
            Option Vanna.

        """
              
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_vanna = (
            (((-carry * d2) / sigma) * nd1) / 100) 

        return opt_vanna               
 

    def vomma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Vomma.

        """
               
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_vomma = (
            (self.vega(S, K, T, r, q, sigma, option, default=False) 
             * ((d1 * d2) / (sigma))) 
            / 100)
        
        return opt_vomma


    def charm(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Charm.

        """
               
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        if option == 'call':
            opt_charm = (
                (-carry * ((nd1 * ((b / (sigma * np.sqrt(T))) 
                                   - (d2 / (2 * T)))) + ((b - r) * Nd1))) 
                / 100)
        if option == 'put':
            opt_charm = (
                (-carry * ((nd1 * ((b / (sigma * np.sqrt(T))) 
                                   - (d2 / (2 * T)))) - ((b - r) * minusNd1))) 
                / 100)
        
        return opt_charm
               

    def zomma(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Zomma.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_zomma = (
            (self.gamma(S, K, T, r, q, sigma, option, default=False) 
            * ((d1 * d2 - 1) / sigma)) 
            / 100)
        
        return opt_zomma


    def speed(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Speed.

        """
        
        if default is None:
            default = True

       # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_speed = -(self.gamma(S, K, T, r, q, sigma, option, default=False) 
                           * (1 + (d1 / (sigma * np.sqrt(T)))) / S)
        
        return opt_speed


    def color(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
              option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Color.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_color = (
            (self.gamma(S, K, T, r, q, sigma, option, default=False) 
             * ((r - b) + ((b * d1) / (sigma * np.sqrt(T))) 
                + ((1 - d1 * d2) / (2 * T)))) 
            / 100)
        
        return opt_color


    def ultima(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
               option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Ultima.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_ultima = (
            (self.vomma(S, K, T, r, q, sigma, option, default=False) 
             * ((1 / sigma) * (d1 * d2 - (d1 / d2) - (d2 / d1) - 1))) 
            / 100)
        
        return opt_ultima


    def vega_bleed(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                   option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Vega Bleed.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 
                'option')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option))
        
        # Update distribution parameters            
        (b, carry, discount, d1, d2, nd1, Nd1, minusNd1, Nd2, 
         minusNd2) = self._refresh_dist(S, K, T, r, q, sigma)
        
        opt_vega_bleed = (
            (self.vega(S, K, T, r, q, sigma, option, default=False) 
             * (r - b + ((b * d1) / (sigma * np.sqrt(T))) 
                - ((1 + (d1 * d2) ) / (2 * T)))) 
            / 100)

        return opt_vega_bleed


    def analytical_sensitivities(self, S=None, K=None, T=None, r=None, q=None, 
                                 sigma=None, option=None, greek=None, 
                                 default=None):
        """
        Sensitivities of the option calculated analytically from closed 
        form solutions.
        
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
            Sensitivity to return. Select from 'delta', 'gamma', 'vega', 
            'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 
            'ultima', 'vega bleed', 'charm'. The default is 'delta'            
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Sensitivity.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option, greek = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'option', 
                'greek')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                    greek=greek))
                    
        for key, value in self.greek_dict.items():
            if str(greek) == key:
                return getattr(self, value)(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                    default=False)
                

    def numerical_delta(self, S=None, K=None, T=None, r=None, q=None, 
                        sigma=None, option=None, price_shift=None, 
                        price_shift_type=None, default=None):
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
            The size of the up and down shift in basis points. The 
            default is 25.
        price_shift_type : Str
            Whether to calculate the change for an upshift, downshift or 
            average of the two. The default is 'avg'.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Delta.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, option, price_shift, 
             price_shift_type) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'option', 'price_shift', 
                'price_shift_type')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                    price_shift=price_shift, 
                    price_shift_type=price_shift_type))
        
        down_shift = S - (price_shift / 10000) * S
        up_shift = S + (price_shift / 10000) * S
        opt_price = self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                               option=option)
        op_shift_down = self.price(S=down_shift, K=K, T=T, r=r, q=q, 
                                   sigma=sigma, option=option)
        op_shift_up = self.price(S=up_shift, K=K, T=T, r=r, q=q, sigma=sigma, 
                                 option=option)
                
        if price_shift_type == 'up':
            opt_delta_shift = (op_shift_up - opt_price) * 4
        if price_shift_type == 'down':
            opt_delta_shift = (opt_price - op_shift_down) * 4
        if price_shift_type == 'avg':    
            opt_delta_shift = (op_shift_up - op_shift_down) * 2
        
        return opt_delta_shift
    
    
    def numerical_sensitivities(self, S=None, K=None, T=None, r=None, q=None, 
                                sigma=None, option=None, greek=None, 
                                price_shift=None, vol_shift=None, 
                                ttm_shift=None, rate_shift=None, default=None):
        """
        Sensitivities of the option calculated numerically using shifts 
        in parameters.

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
            Sensitivity to return. Select from 'delta', 'gamma', 'vega',
            'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 
            'ultima', 'vega bleed', 'charm'. The default is 'delta'
        price_shift : Float
            The size of the price shift in decimal terms. The default 
            is 0.25.
        vol_shift : Float
            The size of the volatility shift in decimal terms. The 
            default is 0.001.
        ttm_shift : Float
            The size of the time to maturity shift in decimal terms. The 
            default is 1/365.  
        rate_shift : Float
            The size of the interest rate shift in decimal terms. The 
            default is 0.0001.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Sensitivity.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, option, greek, price_shift, vol_shift, 
                ttm_shift, rate_shift) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'option', 'greek', 
                'price_shift', 'vol_shift', 'ttm_shift', 
                'rate_shift')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                    greek=greek, price_shift=price_shift, vol_shift=vol_shift, 
                    ttm_shift=ttm_shift, rate_shift=rate_shift))
                    
        if greek in self.equal_greeks:
            option = 'call'
                    
        if greek == 'price':
            result = (
                self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                           default=False))
               
        if greek == 'delta':
            result = (
                (self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                            sigma=sigma, option=option, default=False) 
                 - self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                              sigma=sigma, option=option, default=False)) 
                / (2 * price_shift))
        
        if greek == 'gamma':
            result = (
                (self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                            sigma=sigma, option=option, default=False) 
                 - (2 * self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                                   option=option, default=False)) 
                 + self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                              sigma=sigma, option=option, default=False)) 
                / (price_shift ** 2))
            
        if greek == 'vega':
            result = (
                ((self.price(S=S, K=K, T=T, r=r, q=q, 
                             sigma=(sigma + vol_shift), option=option, 
                             default=False) 
                 - self.price(S=S, K=K, T=T, r=r, q=q, 
                              sigma=(sigma - vol_shift), option=option, 
                              default=False)) 
                 / (2 * vol_shift)) 
                / 100)
        
        if greek == 'theta':
            result = (
                (self.price(S=S, K=K, T=(T - ttm_shift), r=r, q=q, sigma=sigma, 
                            option=option, default=False) 
                 - self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                              option=option, default=False)) 
                / (ttm_shift * 100)) 
        
        if greek == 'rho':
            result = (
                (self.price(S=S, K=K, T=T, r=(r + rate_shift), q=q, 
                            sigma=sigma, option=option, default=False) 
                 - self.price(S=S, K=K, T=T, r=(r - rate_shift), q=q, 
                              sigma=sigma, option=option, default=False)) 
                / (2 * rate_shift * 10000))
                      
        if greek == 'vomma':
            result = (
                ((self.price(S=S, K=K, T=T, r=r, q=q, 
                             sigma=(sigma + vol_shift), option=option, 
                             default=False) 
                  - (2 * self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                                    option=option, default=False)) 
                  + self.price(S=S, K=K, T=T, r=r, q=q, 
                               sigma=(sigma - vol_shift), option=option, 
                               default=False)) 
                 / (vol_shift ** 2)) 
                / 10000)              
        
        if greek == 'vanna':
            result = (
                ((1 / (4 * price_shift * vol_shift)) 
                 * (self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                               sigma=(sigma + vol_shift), option=option, 
                               default=False) 
                    - self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                                 sigma=(sigma - vol_shift), option=option, 
                                 default=False) 
                    - self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                                 sigma=(sigma + vol_shift), option=option, 
                                 default=False) 
                    + self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                                 sigma=(sigma - vol_shift), option=option, 
                                 default=False))) 
                / 100)
        
        if greek == 'charm':
            result = (
                (((self.price(S=(S + price_shift), K=K, T=(T - ttm_shift), r=r, 
                              q=q, sigma=sigma, option=option, default=False) 
                   - self.price(S=(S - price_shift), K=K, T=(T - ttm_shift), 
                                r=r, q=q, sigma=sigma, option=option, 
                                default=False)) 
                  / (2 * price_shift)) 
                 - ((self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                                sigma=sigma, option=option, default=False) 
                     - self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                                  sigma=sigma, option=option, default=False)) 
                    / (2 * price_shift))) 
                / (ttm_shift * 100))
        
        if greek == 'zomma':
            result = (
                ((self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                             sigma=(sigma + vol_shift), option=option, 
                             default=False) 
                  - (2 * self.price(S=S, K=K, T=T, r=r, q=q, 
                                    sigma=(sigma + vol_shift), option=option, 
                                    default=False)) 
                  + self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                               sigma=(sigma + vol_shift), option=option, 
                               default=False)) 
                 - self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                              sigma=(sigma - vol_shift), option=option, 
                              default=False) 
                 + (2 * self.price(S=S, K=K, T=T, r=r, q=q, 
                                   sigma=(sigma - vol_shift), option=option, 
                                   default=False)) 
                 - self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                              sigma=(sigma - vol_shift), option=option, 
                              default=False)) 
                / (2 * vol_shift * (price_shift ** 2)) 
                / 100)
        
        if greek == 'speed':
            result = (
                1 / (price_shift ** 3) 
                * (self.price(S=(S + (2 * price_shift)), K=K, T=T, r=r, q=q, 
                              sigma=sigma, option=option, default=False) 
                   - (3 * self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                                     sigma=sigma, option=option, 
                                     default=False)) 
                   + 3 * self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                                    option=option, default=False) 
                   - self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                                sigma=sigma, option=option, default=False)))
                
        if greek == 'color':
            result = (
                (((self.price(S=(S + price_shift), K=K, T=(T - ttm_shift), 
                              r=r, q=q, sigma=sigma, option=option, 
                              default=False) 
                   - (2 * self.price(S=S, K=K, T=(T - ttm_shift), r=r, q=q, 
                                     sigma=sigma, option=option, 
                                     default=False)) 
                   + self.price(S=(S - price_shift), K=K, T=(T - ttm_shift), 
                                r=r, q=q, sigma=sigma, option=option, 
                                default=False)) 
                  / (price_shift ** 2)) 
                 - ((self.price(S=(S + price_shift), K=K, T=T, r=r, q=q, 
                                sigma=sigma, option=option, default=False) 
                     - (2 * self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                                       option=option, default=False)) 
                     + self.price(S=(S - price_shift), K=K, T=T, r=r, q=q, 
                                  sigma=sigma, option=option, default=False)) 
                    / (price_shift ** 2) )) 
                / (ttm_shift * 100))
            
        if greek == 'ultima':
            result = (
                (1 / (vol_shift ** 3) 
                 * (self.price(S=S, K=K, T=T, r=r, q=q, 
                               sigma=(sigma + (2 * vol_shift)), 
                               option=option, default=False) 
                    - (3 * self.price(S=S, K=K, T=T, r=r, q=q, 
                                      sigma=(sigma + vol_shift), 
                                      option=option, default=False)) 
                    + 3 * self.price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                                     option=option, default=False) 
                    - self.price(S=S, K=K, T=T, r=r, q=q, 
                                 sigma=(sigma - vol_shift), 
                                 option=option, default=False))) 
                * (vol_shift ** 2))
        
        if greek == 'vega bleed':
            result = (
                (((self.price(S=S, K=K, T=(T - ttm_shift), r=r, q=q, 
                              sigma=(sigma + vol_shift), option=option, 
                              default=False) 
                   - self.price(S=S, K=K, T=(T - ttm_shift), r=r, q=q, 
                                sigma=(sigma - vol_shift), 
                                option=option, default=False)) 
                  / (2 * vol_shift)) 
                 - ((self.price(S=S, K=K, T=T, r=r, q=q, 
                                sigma=(sigma + vol_shift), 
                                option=option, default=False) 
                     - self.price(S=S, K=K, T=T, r=r, q=q, 
                                  sigma=(sigma - vol_shift), 
                                  option=option, default=False)) 
                    / (2 * vol_shift))) 
                / (ttm_shift * 10000))
        
        return result


    def sensitivities(self, S=None, K=None, T=None, r=None, q=None, 
                      sigma=None, option=None, greek=None, price_shift=None, 
                      vol_shift=None, ttm_shift=None, rate_shift=None, 
                      num_sens=None, default=None):
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
            Sensitivity to return. Select from 'delta', 'gamma', 'vega', 
            'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 'color', 
            'ultima', 'vega bleed', 'charm'. The default is 'delta'
        price_shift : Float
            The size of the price shift in decimal terms. The default 
            is 0.25.
        vol_shift : Float
            The size of the volatility shift in decimal terms. The 
            default is 0.001.
        ttm_shift : Float
            The size of the time to maturity shift in decimal terms. The 
            default is 1/365.
        rate_shift : Float
            The size of the interest rate shift in decimal terms. The 
            default is 0.0001.    
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. 
            The default is False.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Option Sensitivity.

        """

        if num_sens is None:
            num_sens = self.df_dict['df_num_sens']
        
        if num_sens:
            return self.numerical_sensitivities(
                S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                greek=greek, price_shift=price_shift, vol_shift=vol_shift, 
                ttm_shift=ttm_shift, rate_shift=rate_shift, default=default)            
            
        else:
            return self.analytical_sensitivities(
                S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                greek=greek, default=default)


    def barrier_price(self, S=None, K=None, H=None, R=None, T=None, r=None, 
                      q=None, sigma=None, barrier_direction=None, knock=None, 
                      option=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or used within a graph call where they have 
            already been updated.

        Returns
        -------
        Float
            Barrier option price.

        """

        if default is None:
            default = True
               
        # If default is set to False the price is to be used in combo 
        # graphs so the distributions are refreshed but not the 
        # parameters.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, H, R, T, r, q, sigma, option, barrier_direction, 
             knock) = itemgetter(
                 'S', 'K', 'H', 'R', 'T', 'r', 'q', 'sigma', 'option', 
                 'barrier_direction', 'knock')(self._refresh_params_default(
                     S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                     option=option, barrier_direction=barrier_direction, 
                     knock=knock))
               
                     
        # Down and In Call
        if (barrier_direction == 'down' 
            and knock == 'in' 
            and option == 'call'):
            
            self.eta = 1
            self.phi = 1
        
            A, B, C, D, E, F = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
                   
            if K > H:
                opt_barrier_payoff = C + E
            if K < H:
                opt_barrier_payoff = A - B + D + E
            

        # Up and In Call    
        if (barrier_direction == 'up' 
                and knock == 'in' 
                and option == 'call'):
            
            self.eta = -1
            self.phi = 1
            
            (A, B, C, D, E, F) = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
            
            if K > H:
                opt_barrier_payoff = A + E
            if K < H:
                opt_barrier_payoff = B - C + D + E


        # Down and In Put
        if (barrier_direction == 'down' 
                and knock == 'in' 
                and option == 'put'):

            self.eta = 1
            self.phi = -1
            
            (A, B, C, D, E, F) = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
            
            if K > H:
                opt_barrier_payoff = B - C + D + E
            if K < H:
                opt_barrier_payoff = A + E
                
                
        # Up and In Put         
        if (barrier_direction == 'up' 
            and knock == 'in' 
            and option == 'put'):
            
            self.eta = -1
            self.phi = -1
            
            (A, B, C, D, E, F) = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
        
            if K > H:
                opt_barrier_payoff = A - B + D + E
            if K < H:
                opt_barrier_payoff = C + E
                
                
        # Down and Out Call
        if (barrier_direction == 'down' 
            and knock == 'out' 
            and option == 'call'):
            
            self.eta = 1
            self.phi = 1
            
            (A, B, C, D, E, F) = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
        
            if K > H:
                opt_barrier_payoff = A - C + F
            if K < H:
                opt_barrier_payoff = B - D + F
            
            
        # Up and Out Call
        if (barrier_direction == 'up' 
            and knock == 'out' 
            and option == 'call'):
            
            self.eta = -1
            self.phi = 1
            
            (A, B, C, D, E, F) = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
            
            if K > H:
                opt_barrier_payoff = F
            if K < H:
                opt_barrier_payoff = (A - B + C 
                                           - D + F)


        # Down and Out Put
        if (barrier_direction == 'down' 
            and knock == 'out' 
            and option == 'put'):
            
            self.eta = 1
            self.phi = -1
            
            (A, B, C, D, E, F) = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
            
            if K > H:
                opt_barrier_payoff = (A - B + C 
                                           - D + F)
            if K < H:
                opt_barrier_payoff = F
                
        # Up and Out Put         
        if (barrier_direction == 'up' 
            and knock == 'out' 
            and option == 'put'):
            
            self.eta = -1
            self.phi = -1
            
            (A, B, C, D, E, F) = itemgetter(
                'A', 'B', 'C', 'D', 'E', 'F')(self._barrier_factors(
                    S=S, K=K, H=H, R=R, T=T, r=r, q=q, sigma=sigma, 
                    phi=self.phi, eta=self.eta))
        
            if K > H:
                opt_barrier_payoff = B - D + F
            if K < H:
                opt_barrier_payoff = A - C + F

        return opt_barrier_payoff    


    def visualize(
            self, risk=None, S=None, T=None, r=None, q=None, sigma=None, 
            option=None, direction=None, greek=None, graphtype=None, 
            x_plot=None, y_plot=None, G1=None, G2=None, G3=None, T1=None, 
            T2=None, T3=None, time_shift=None, interactive=None, notebook=None, 
            colorscheme=None, colorintensity=None, size2d=None, size3d=None, 
            axis=None, spacegrain=None, azim=None, elev=None, K=None, K1=None, 
            K2=None, K3=None, K4=None, cash=None, ratio=None, value=None, 
            combo_payoff=None, mpl_style=None, num_sens=None, gif=None):
        """
        Plot the chosen graph of risk or payoff.
        
        
        Parameters
        ----------
        risk : Bool
            Whether to display risk graph or payoff graph. The default 
            is True.
        S : Float
            Underlying Stock Price. Used in risk & payoff graphs. The 
            default is 100. 
        T : Float
            Time to Maturity. Used in risk & payoff graphs. The default 
            is 0.25 (3 months).
        r : Float
            Interest Rate. Used in risk & payoff graphs. The default 
            is 0.05 (5%).
        q : Float
            Dividend Yield. Used in risk & payoff graphs. The default 
            is 0.
        sigma : Float
            Implied Volatility. Used in risk & payoff graphs. The default 
            is 0.2 (20%).
        option : Str
            Option type, Put or Call. Used in risk & payoff graphs. The 
            default is 'call'
        direction : Str
            Whether the payoff is long or short. Used in risk & payoff 
            graphs. The default is 'long'.
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 
            'color', 'ultima', 'vega_bleed', 'charm'. Used in risk 
            graphs. The default is 'delta'. 
        graphtype : Str
            Whether to plot 2D or 3D graph. Used in risk graphs. The 
            default is 2D.
        x_plot : Str
            The x-axis variable - 'price', 'strike', 'vol' or 'time'. 
            Used in 2D-risk graphs. The default is 'price'.
        y_plot : Str
            The y-axis variable - 'value', 'delta', 'gamma', 'vega' or 
            'theta'. Used in 2D-risk graphs. The default is 'delta.
        G1 : Float
            Strike Price of option 1. Used in 2D-risk graphs. The 
            default is 90.
        G2 : Float
            Strike Price of option 2. Used in 2D-risk graphs. The 
            default is 100.
        G3 : Float
            Strike Price of option 3. Used in 2D-risk graphs. The 
            default is 110.
        T1 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The 
            default is 0.25 (3 months).
        T2 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The 
            default is 0.25 (3 months).
        T3 : Float
            Time to Maturity of option 1. Used in 2D-risk graphs. The 
            default is 0.25 (3 months).
        time_shift : Float
            Difference between T1 and T2 in rho graphs. Used in 2D-risk 
            graphs. The default is 0.25 (3 months).
        interactive : Bool
            Whether to show matplotlib (False) or plotly (True) graph. 
            Used in 3D-risk graphs. The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and 
            hence whether it should output in line or to an HTML file. 
            Used in 3D-risk graphs. The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. Used 
            in 3D-risk graphs. The default is 'jet' (which is a palette 
            that works in both plotly and matplotlib).
        colorintensity : Float
            The alpha value indicating level of transparency / 
            intensity. The default is 1.
        size2d : Tuple
            Figure size for matplotlib chart. Used in 2D-risk & payoff 
            graphs. The default is (6, 4).    
        size3d : Tuple
            Figure size for matplotlib chart. Used in 3D-risk graphs. 
            The default is (15, 12).
        axis : Str
            Whether the x-axis is 'price' or 'vol'. Used in 3D-risk 
            graphs. The default is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D 
            graphs. Used in 3D-risk graphs. The default is 100. 
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.      
        K : Float
            Strike Price of option 1. Used in payoff graphs. The 
            default is 100 (individual payoffs may have own defaults).
        K1 : Float
             Strike Price of option 1. Used in payoff graphs. The 
             default is 95 (individual payoffs may have own defaults).
        K2 : Float
             Strike Price of option 2. Used in payoff graphs. The 
             default is 105 (individual payoffs may have own defaults).
        K3 : Float
             Strike Price of option 3. Used in payoff graphs. The 
             default is 105 (individual payoffs may have own defaults).
        K4 : Float
             Strike Price of option 4. Used in payoff graphs. The 
             default is 105 (individual payoffs may have own defaults).
        cash : Bool
            Whether to discount forward to present value. Used in 
            forward payoff graph. The default is False.
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. Used 
            in backspread, ratio vertical spread payoff graphs. The 
            default is 2. 
        value : Bool
            Whether to show the current value as well as the terminal 
            payoff. Used in payoff graphs. The default is False.
        combo_payoff : Str
            The payoff to be displayed. Used in payoff graphs. The 
            default is 'straddle'.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. 
            The default is False.
        gif : Bool    
            Whether to create an animated gif. The default is False.
            
        Returns
        -------
        Displays graph of either 2D / 3D greeks or payoff diagram.

        """
        
        if risk is None:
            risk = self.df_dict['df_risk']
        
        if risk:
            self.greeks(
                x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, G3=G3, T=T, 
                T1=T1, T2=T2, T3=T3, time_shift=time_shift, r=r, q=q, 
                sigma=sigma, option=option, direction=direction, 
                interactive=interactive, notebook=notebook, 
                colorscheme=colorscheme, colorintensity=colorintensity, 
                size2d=size2d, size3d=size3d, axis=axis, spacegrain=spacegrain, 
                azim=azim, elev=elev, greek=greek, graphtype=graphtype, 
                mpl_style=mpl_style, num_sens=num_sens, gif=gif)
        
        else:
            self.payoffs(
                S=S, K=K, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                sigma=sigma, option=option, direction=direction, size2d=size2d, 
                cash=cash, ratio=ratio, value=value, combo_payoff=combo_payoff, 
                mpl_style=mpl_style)
        
    
    def greeks(self, x_plot=None, y_plot=None, S=None, G1=None, G2=None, 
               G3=None, T=None, T1=None, T2=None, T3=None, time_shift=None, 
               r=None, q=None, sigma=None, option=None, direction=None, 
               interactive=None, notebook=None, colorscheme=None, 
               colorintensity=None, size2d=None, size3d=None, axis=None, 
               spacegrain=None, azim=None, elev=None, greek=None, 
               graphtype=None, mpl_style=None, num_sens=None, gif=None):
        """
        Plot the chosen 2D or 3D graph
        
        
        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 
            'color', 'ultima', 'vega_bleed', 'charm'. The default is 
            'delta'
        x_plot : Str
                 The x-axis variable ('price', 'strike', 'vol' or 
                 'time'). The default is 'price'.
        y_plot : Str
                 The y-axis variable ('value', 'delta', 'gamma', 'vega' 
                 or 'theta'). The default is 'delta.
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
             Time to Maturity of option 1. The default is 0.25 
             (3 months).
        T2 : Float
             Time to Maturity of option 1. The default is 0.25 
             (3 months).
        T3 : Float
             Time to Maturity of option 1. The default is 0.25 
             (3 months).
        time_shift : Float
             Difference between T1 and T2 in rho graphs. The default 
             is 0.25 (3 months).
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
            Whether to show matplotlib (False) or plotly (True) graph. 
            The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and 
            hence whether it should output in line or to an HTML file. 
            The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. The 
            default is 'jet' (which works in both).
        colorintensity : Float
            The alpha value indicating level of transparency / 
            intensity. The default is 1.
        size2d : Tuple
            Figure size for matplotlib chart. The default is (6, 4).
        size3d : Tuple
            Figure size for matplotlib chart. The default is (15, 12).    
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default is 
            'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D 
            graphs. The default is 100.  
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.      
        graphtype : Str
            Whether to plot 2D or 3D graph. The default is 2D.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'. 
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. 
            The default is False.
        gif : Bool    
            Whether to create an animated gif. The default is False.    

        Returns
        -------
        Runs method to display either 2D or 3D greeks graph.

        """
        
        if graphtype is None:
            graphtype = self.df_dict['df_graphtype']
        
        # Run 2D greeks method
        if graphtype == '2D':
            self.greeks_graphs_2D(
                x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, G3=G3, T=T, 
                T1=T1, T2=T2, T3=T3, time_shift=time_shift, r=r, q=q, 
                sigma=sigma, option=option, direction=direction, 
                size2d=size2d, mpl_style=mpl_style, num_sens=num_sens)
        
        # Run 3D greeks method    
        if graphtype == '3D':
            self.greeks_graphs_3D(
                S=S, r=r, q=q, sigma=sigma, option=option, 
                interactive=interactive, notebook=notebook, 
                colorscheme=colorscheme, colorintensity=colorintensity, 
                size3d=size3d, direction=direction, axis=axis, 
                spacegrain=spacegrain, azim=azim, elev=elev, greek=greek, 
                num_sens=num_sens, gif=gif)
    
    
    def greeks_graphs_2D(self, x_plot=None, y_plot=None, S=None, G1=None, 
                         G2=None, G3=None, T=None, T1=None, T2=None, T3=None, 
                         time_shift=None, r=None, q=None, sigma=None, 
                         option=None, direction=None, size2d=None, 
                         mpl_style=None, num_sens=None, gif=None):
        """
        Plot chosen 2D greeks graph.
                

        Parameters
        ----------
        x_plot : Str
                 The x-axis variable ('price', 'strike', 'vol' or 
                 'time'). The default is 'price'.
        y_plot : Str
                 The y-axis variable ('value', 'delta', 'gamma', 'vega' 
                 or 'theta'). The default is 'delta.
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
             Time to Maturity of option 1. The default is 0.25 
             (3 months).
        T2 : Float
             Time to Maturity of option 1. The default is 0.25 
             (3 months).
        T3 : Float
             Time to Maturity of option 1. The default is 0.25 
             (3 months).
        time_shift : Float
             Difference between T1 and T2 in rho graphs. The default 
             is 0.25 (3 months).
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
        size2d : Tuple
            Figure size for matplotlib chart. The default is (6, 4).     
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'. 
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. 
            The default is False.    

        Returns
        -------
        Runs method to create data for 2D greeks graph.

        """
                
        # Pass parameters to be initialised. If not provided they will 
        # be populated with default values
        (x_plot, y_plot, S, G1, G2, G3, T, T1, T2, T3, time_shift, r, q, 
         sigma, option, direction, size2d, mpl_style, num_sens, 
         gif) = itemgetter(
            'x_plot', 'y_plot', 'S', 'G1', 'G2', 'G3', 'T', 'T1', 'T2', 'T3', 
            'time_shift', 'r', 'q', 'sigma', 'option', 'direction', 'size2d', 
            'mpl_style', 'num_sens', 'gif')(self._refresh_params_default(
                x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, G3=G3, T=T, 
                T1=T1, T2=T2, T3=T3, time_shift=time_shift, r=r, q=q, 
                sigma=sigma, option=option, direction=direction, size2d=size2d, 
                mpl_style=mpl_style, num_sens=num_sens, gif=gif))
            
        if gif:
            fig, ax = self._2D_general_graph(
                x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, G3=G3, T=T, 
                T1=T1, T2=T2, T3=T3, time_shift=time_shift, r=r, q=q, 
                sigma=sigma, option=option, direction=direction, size2d=size2d, 
                mpl_style=mpl_style, num_sens=num_sens, gif=gif)
            
            return fig, ax
            
        else:
            self._2D_general_graph(
                x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, G3=G3, T=T, 
                T1=T1, T2=T2, T3=T3, time_shift=time_shift, r=r, q=q, 
                sigma=sigma, option=option, direction=direction, size2d=size2d, 
                mpl_style=mpl_style, num_sens=num_sens, gif=gif)       
    

    def _2D_general_graph(self, x_plot, y_plot, S, G1, G2, G3, T, T1, T2, T3, 
                          time_shift, r, q, sigma, option, direction, size2d, 
                          mpl_style, num_sens, gif):                               
        """
        Creates data for 2D greeks graph.

        Returns
        -------
        Runs method to graph using Matplotlib.

        """
        
        # create arrays of 1000 equally spaced points for a range of 
        # strike prices, volatilities and maturities
        self.SA = np.linspace(0.8 * S, 1.2 * S, 1000)
        self.sigmaA = np.linspace(0.05, 0.5, 1000)
        self.TA = np.linspace(0.01, 1, 1000)
        
        # y-axis parameters other than rho require 3 options to be 
        # graphed
        if y_plot in self.y_name_dict.keys():
            for opt in [1, 2, 3]:
                if x_plot == 'price':
                    
                    # Use self.__dict__ to access names, C1... etc., 
                    # For price we set S to the array SA 
                    self.__dict__[
                        'C'+str(opt)] = self.sensitivities(
                            S=self.SA, K=self.__dict__['G'+str(opt)], 
                            T=self.__dict__['T'+str(opt)], r=r, q=q, 
                            sigma=sigma, option=option, 
                            greek=self.y_name_dict[y_plot], 
                            price_shift=0.25, vol_shift=0.001, 
                            ttm_shift=(1 / 365), rate_shift=0.0001, 
                            num_sens=num_sens, default=False)        
                            
                if x_plot == 'vol':
                    
                    # For vol we set sigma to the array sigmaA
                    self.__dict__[
                        'C'+str(opt)] = self.sensitivities(
                            S=S, K=self.__dict__['G'+str(opt)], 
                            T=self.__dict__['T'+str(opt)], r=r, q=q, 
                            sigma=self.sigmaA, option=option, 
                            greek=self.y_name_dict[y_plot], 
                            price_shift=0.25, vol_shift=0.001, 
                            ttm_shift=(1 / 365), rate_shift=0.0001, 
                            num_sens=num_sens, default=False)        
                            
                if x_plot == 'time':
                    
                    # For time we set T to the array TA
                    self.__dict__[
                        'C'+str(opt)] = self.sensitivities(
                            S=S, K=self.__dict__['G'+str(opt)], T=self.TA, r=r, 
                            q=q, sigma=sigma, option=option, 
                            greek=self.y_name_dict[y_plot], 
                            price_shift=0.25, vol_shift=0.001, 
                            ttm_shift=(1 / 365), rate_shift=0.0001, 
                            num_sens=num_sens, default=False)
                    
            
            # Reverse the option value if direction is 'short'        
            if direction == 'short':
                for opt in [1, 2, 3]:
                    self.__dict__['C'+str(opt)] = -self.__dict__['C'+str(opt)]
            
            # Call strike_tenor_label method to assign labels to chosen 
            # strikes and tenors
            self._strike_tenor_label()
 
        # rho requires 4 options to be graphed 
        if y_plot == 'rho':
            
            # Set T1 and T2 to the specified time and shifted time
            self.T1 = T
            self.T2 = T + time_shift
            
            # 2 Tenors
            tenor_type = {1:1, 2:2, 3:1, 4:2}
            
            # And call and put for each tenor 
            opt_type = {1:'call', 2:'call', 3:'put', 4:'put'}
            for opt in [1, 2, 3, 4]:
                if x_plot == 'price':
                    
                    # For price we set S to the array SA
                    self.__dict__[
                        'C'+str(opt)] = self.sensitivities(
                            S=self.SA, K=G2, 
                            T=self.__dict__['T'+str(tenor_type[opt])], r=r, 
                            q=q, sigma=sigma, option=opt_type[opt], 
                            greek=y_plot, price_shift=0.25, 
                            vol_shift=0.001, ttm_shift=(1 / 365), 
                            rate_shift=0.0001, num_sens=num_sens, 
                            default=False)
                           
                if x_plot == 'strike':
                    
                    # For strike we set K to the array SA
                    self.__dict__[
                        'C'+str(opt)] = self.sensitivities(
                            S=S, K=self.SA, 
                            T=self.__dict__['T'+str(tenor_type[opt])], r=r, 
                            q=q, sigma=sigma, option=opt_type[opt], 
                            greek=y_plot, price_shift=0.25, 
                            vol_shift=0.001, ttm_shift=(1 / 365), 
                            rate_shift=0.0001, num_sens=num_sens, 
                            default=False)
                            
                if x_plot == 'vol':
                    
                    # For vol we set sigma to the array sigmaA
                    self.__dict__[
                        'C'+str(opt)] = self.sensitivities(
                            S=S, K=G2, 
                            T=self.__dict__['T'+str(tenor_type[opt])], r=r, 
                            q=q, sigma=self.sigmaA, option=opt_type[opt], 
                            greek=y_plot, price_shift=0.25, 
                            vol_shift=0.001, ttm_shift=(1 / 365), 
                            rate_shift=0.0001, num_sens=num_sens, 
                            default=False)
            
            # Reverse the option value if direction is 'short'        
            if direction == 'short':
                for opt in [1, 2, 3, 4]:
                    self.__dict__['C'+str(opt)] = -self.__dict__['C'+str(opt)]
    
            # Assign the option labels
            self.label1 = str(int(self.T1 * 365))+' Day Call'
            self.label2 = str(int(self.T2 * 365))+' Day Call'
            self.label3 = str(int(self.T1 * 365))+' Day Put'
            self.label4 = str(int(self.T2 * 365))+' Day Put'
    
        # Convert the x-plot and y-plot values to axis labels   
        xlabel = self.label_dict[str(x_plot)]
        ylabel = self.label_dict[str(y_plot)]
        
        # If the greek is rho or the same for a call or a put, set the 
        # option name to 'Call / Put' 
        if y_plot in [self.equal_greeks, 'rho']:
                option = 'Call / Put'     
        
        # Create chart title as direction plus option type plus y-plot 
        # vs x-plot    
        title = (str(direction.title())+' '+str(option.title())
                 +' '+y_plot.title()+' vs '+x_plot.title())   
        
        # Set the x-axis array as price, vol or time
        x_name = str(x_plot)
        if x_name in self.x_name_dict.keys():
            xarray = (self.__dict__[str(self.x_name_dict[x_name])] * 
                      self.x_scale_dict[x_name])
        
        # Plot 3 option charts    
        if y_plot in self.y_name_dict.keys():        
            if gif:
                fig, ax = self._vis_greeks_mpl(
                    x_plot=x_plot, yarray1=self.C1, yarray2=self.C2, 
                    yarray3=self.C3, xarray=xarray, label1=self.label1, 
                    label2=self.label2, label3=self.label3, xlabel=xlabel, 
                    ylabel=ylabel, title=title, size2d=size2d, mpl_style=mpl_style, 
                    gif=gif)
                return fig, ax
            
            else:
                self._vis_greeks_mpl(
                    x_plot=x_plot, yarray1=self.C1, yarray2=self.C2, 
                    yarray3=self.C3, xarray=xarray, label1=self.label1, 
                    label2=self.label2, label3=self.label3, xlabel=xlabel, 
                    ylabel=ylabel, title=title, size2d=size2d, mpl_style=mpl_style, 
                    gif=gif)
        
        # Plot Rho charts    
        elif y_plot == 'rho':
            self._vis_greeks_mpl(
                x_plot=x_plot, yarray1=self.C1, yarray2=self.C2, 
                yarray3=self.C3, yarray4=self.C4, xarray=xarray, 
                label1=self.label1, label2=self.label2, label3=self.label3, 
                label4=self.label4, xlabel=xlabel, ylabel=ylabel, title=title, 
                size2d=size2d, mpl_style=mpl_style, gif=False)
 
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
        strike_label = dict()
        for key, value in {'G1':'label1', 'G2':'label2', 
                           'G3':'label3'}.items():
            
            # If the strike is 100% change name to 'ATM'
            if self.__dict__[str(key)] == self.S:
                strike_label[value] = 'ATM Strike'
            else:
                strike_label[value] = str(int(
                    self.__dict__[key]))+' Strike' 
               
        for k, v in {'T1':'label1', 'T2':'label2', 'T3':'label3'}.items():
            
            # Make each label value the number of days to maturity 
            # plus the strike level
            self.__dict__[v] = str(
                int(self.__dict__[str(k)]*365))+' Day '+strike_label[str(v)]
                
        return self                           


    def _vis_greeks_mpl(self, x_plot, xarray, yarray1, yarray2, yarray3, 
                        label1, label2, label3, xlabel, ylabel, title, size2d, 
                        mpl_style, gif, yarray4=None, label4=None):
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
        
        # Set style to chosen mpl_style (default is Seaborn Darkgrid)
        plt.style.use(mpl_style)

        # Update chart parameters        
        pylab.rcParams.update(self.mpl_params)
        
        # Create the figure and axes objects
        fig, ax = plt.subplots(figsize=size2d)
        
        # If plotting against time, show time to maturity reducing left 
        # to right
        if x_plot == 'time':
            ax.invert_xaxis()
            
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
        
        # Apply a black border to the chart
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('1')
        
        # Set x and y axis labels and title
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
 
        # Create a legend 
        ax.legend(loc=0, fontsize=10)
        
        if gif:
            return fig, ax
        
        else:
            # Display the chart
            plt.show()
    
    
    def greeks_graphs_3D(self, S=None, r=None, q=None, sigma=None, 
                         option=None, interactive=None, notebook=None, 
                         colorscheme=None, colorintensity=None, size3d=None, 
                         direction=None, axis=None, spacegrain=None, azim=None,
                         elev=None, greek=None, num_sens=None, gif=None):
        """
        Plot chosen 3D greeks graph.

        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 
            'color', 'ultima', 'vega_bleed', 'charm'. The default is 
            'delta'
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
            Whether to show matplotlib (False) or plotly(True) graph. 
            The default is False.
        notebook : Bool
            Whether the function is being run in an IPython notebook and 
            hence whether it should output in line or to an HTML file. 
            The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. The 
            default is 'jet' (which works in both).
        colorintensity : Float
            The alpha value indicating level of transparency / 
            intensity. The default is 1.
        size3d : Tuple
            Figure size for matplotlib chart. The default is (15, 12).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default 
            is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D 
            graphs. The default is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.      
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. 
            The default is False. 
        gif : Bool    
            Whether to create an animated gif. The default is False.    

        Returns
        -------
        Runs method to display 3D greeks graph.

        """
        
        # Pass parameters to be initialised. If not provided they will 
        # be populated with default values
        (S, r, q, sigma, option, interactive, notebook, colorscheme, 
         colorintensity, size3d, azim, elev, direction, axis, spacegrain, 
         greek, num_sens, gif) = itemgetter(
            'S', 'r', 'q', 'sigma', 'option', 'interactive', 'notebook', 
            'colorscheme', 'colorintensity', 'size3d', 'azim', 'elev', 
            'direction', 'axis', 'spacegrain', 'greek', 
            'num_sens', 'gif')(self._refresh_params_default(
                S=S, r=r, q=q, sigma=sigma, option=option, 
                interactive=interactive, notebook=notebook, 
                colorscheme=colorscheme, colorintensity=colorintensity, 
                size3d=size3d, azim=azim, elev=elev, direction=direction, 
                axis=axis, spacegrain=spacegrain, greek=greek, 
                num_sens=num_sens, gif=gif))
               
        # Select the input name and method name from the greek 
        # dictionary 
        for greek_label, greek_func in self.greek_dict.items():
            
            # If the greek is the same for call or put, set the option 
            # value to 'Call / Put'
            if greek in self.equal_greeks:
                option = 'Call / Put'
            
            # For the specified greek
            if greek == greek_label:

                # Prepare the graph axes                 
                (x, y, xmin, xmax, ymin, ymax, graph_scale, axis_label1, 
                 axis_label2) = self._graph_space_prep(
                     greek=greek, S=S, axis=axis, spacegrain=spacegrain)
                     
                if axis == 'price':
                    
                    # Select the individual greek method from sensitivities
                    z = self.sensitivities(
                        S=x, K=S, T=y, r=r, q=q, sigma=sigma, option=option, 
                        greek=greek, price_shift=0.25, vol_shift=0.001, 
                        ttm_shift=(1 / 365), num_sens=num_sens, default=False)
               
                if axis == 'vol':
                    
                    # Select the individual greek method from sensitivities
                    z = self.sensitivities(
                        S=S, K=S, T=y, r=r, q=q, sigma=x, option=option, 
                        greek=greek, price_shift=0.25, vol_shift=0.001, 
                        ttm_shift=(1 / 365), num_sens=num_sens, default=False)
        
        # Run the 3D visualisation method            
        if gif:
            fig, ax, titlename, title_font_scale = self._vis_greeks_3D(
                x, y, z, xmin, xmax, ymin, ymax, graph_scale, axis_label1, 
                axis_label2, direction, option, greek, interactive, 
                colorscheme, colorintensity, size3d, azim, elev, notebook, gif)
            return fig, ax, titlename, title_font_scale
        
        else:
            self._vis_greeks_3D(
                x, y, z, xmin, xmax, ymin, ymax, graph_scale, axis_label1, 
                axis_label2, direction, option, greek, interactive, 
                colorscheme, colorintensity, size3d, azim, elev, notebook, gif)            
    
    
    def _graph_space_prep(self, greek, S, axis, spacegrain):
        """
        Prepare the axis ranges to be used in 3D graph.

        Parameters
        ----------
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 
            'color', 'ultima', 'vega_bleed', 'charm'. The default is 
            'delta'
        S : Float
             Underlying Stock Price. The default is 100.
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default 
            is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D 
            graphs. The default is 100.

        Returns
        -------
        Various
            Updated parameters to be used in 3D graph.

        """
        
        # Select the strike and Time ranges for each greek from the 3D 
        # chart ranges dictionary 
        SA_lower = self.df_dict['df_3D_chart_ranges'][str(greek)]['SA_lower']
        SA_upper = self.df_dict['df_3D_chart_ranges'][str(greek)]['SA_upper']
        TA_lower = self.df_dict['df_3D_chart_ranges'][str(greek)]['TA_lower']
        TA_upper = self.df_dict['df_3D_chart_ranges'][str(greek)]['TA_upper']
        
        # Set the volatility range from 5% to 50%
        sigmaA_lower = 0.05 
        sigmaA_upper = 0.5 

        # create arrays of 100 equally spaced points for the ranges of 
        # strike prices, volatilities and maturities
        SA = np.linspace(SA_lower * S, SA_upper * S, int(spacegrain))
        TA = np.linspace(TA_lower, TA_upper, int(spacegrain))
        sigmaA = np.linspace(sigmaA_lower, sigmaA_upper, int(spacegrain))
        
        # set y-min and y-max labels 
        ymin = TA_lower
        ymax = TA_upper
        axis_label2 = 'Time to Expiration (Days)'
        
        # set x-min and x-max labels 
        if axis == 'price':
            x, y = np.meshgrid(SA, TA)
            xmin = SA_lower
            xmax = SA_upper
            graph_scale = 1
            axis_label1 = 'Underlying Value'            
            
        if axis == 'vol':
            x, y = np.meshgrid(sigmaA, TA)
            xmin = sigmaA_lower
            xmax = sigmaA_upper    
            graph_scale = 100
            axis_label1 = 'Volatility %'    

        return (x, y, xmin, xmax, ymin, ymax, graph_scale, axis_label1, 
                axis_label2)
    
    
    def _titlename(self, option, direction, greek):
        """
        Create graph title based on option type, direction and greek

        Returns
        -------
        Graph title.

        """
        # Label the graph based on whether it is different for calls 
        # & puts or the same
        if option == 'Call / Put':
            titlename = str(str(direction.title())+' '+option
                            +' Option '+str(greek.title()))
        else:    
            titlename = str(str(direction.title())+' '
                            +str(option.title())+' Option '
                            +str(greek.title()))    
        return titlename
    
    
    def _plotly_3D_ranges(self, x, y, z, xmin, xmax, ymin, ymax, graph_scale):
        """
        Generate contour ranges and format axes for plotly 3D graph

        Returns
        -------
        axis ranges
        
        """
        # Set the ranges for the contour values and reverse / rescale axes
        x, y = y * 365, x * graph_scale
        x_start = ymin
        x_stop = ymax * 360
        x_size = x_stop / 18
        y_start = xmin
        y_stop = xmax * graph_scale
        y_size = int((xmax - xmin) / 20)
        z_start = np.min(z)
        z_stop = np.max(z)
        z_size = int((np.max(z) - np.min(z)) / 10)
        
        return (x, y, x_start, x_stop, x_size, y_start, y_stop, y_size, 
                z_start, z_stop, z_size)
               
    
    def _plotly_3D(
            self, x, y, z, x_start, x_stop, x_size, y_start, y_stop, y_size, 
            z_start, z_stop, z_size, colorscheme, titlename, axis_label1, 
            axis_label2, axis_label3, notebook):
        """
        Display 3D greeks graph.

        Returns
        -------
        plotly 3D graph

        """
        # create plotly figure object
        fig = go.Figure(
            data=[go.Surface(x=x, 
                             y=y, 
                             z=z, 

                             # set the colorscale to the chosen 
                             # colorscheme
                             colorscale=colorscheme, 
                            
                             # Define the contours
                             contours = {"x": {"show": True, 
                                               "start": x_start, 
                                               "end": x_stop, 
                                               "size": x_size, 
                                               "color":"white"},            
                                         "y": {"show": True, 
                                               "start": y_start, 
                                               "end": y_stop, 
                                               "size": y_size, 
                                               "color":"white"},  
                                         "z": {"show": True, 
                                               "start": z_start, 
                                               "end": z_stop, 
                                               "size": z_size}},)])
        
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
                            xaxis_title=axis_label2,
                            yaxis_title=axis_label1,
                            zaxis_title=axis_label3,),
                          title={'text':titlename,
                                 'y':0.9,
                                 'x':0.5,
                                 'xanchor':'center',
                                 'yanchor':'top',
                                 'font':dict(size=20, 
                                             color="black")},
                          autosize=False, 
                          width=800, height=800,
                          margin=dict(l=65, r=50, b=65, t=90),
                          scene_camera=camera)
        
        if notebook is None:
            notebook = self.df_dict['df_notebook']
        
        # If running in an iPython notebook the chart will display 
        # in line
        if notebook:
            fig.show()
        
        # Otherwise create an HTML file that opens in a new window
        else:
            plot(fig, auto_open=True)
    
    
    def _mpl_axis_format(self, x, y, graph_scale):
        """
        Rescale Matplotlib axis values

        Returns
        -------
        x, y axis values

        """
        x = x * graph_scale
        y = y * 365
        return x, y
        
    
    def _mpl_3D(self, x, y, z, size3d, azim, elev, axis_label1, axis_label2, 
                axis_label3, colorscheme, colorintensity, titlename, gif):
        """
        Display 3D greeks graph.

        Returns
        -------
        Matplotlib static graph.

        """
                
        # Update chart parameters        
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update(self.mpl_3d_params)
        
        # create figure with specified size tuple
        fig = plt.figure(figsize=size3d)
        ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev)
        
        # Set background color to white
        ax.set_facecolor('w')

        # Create values that scale fonts with fig_size 
        ax_font_scale = int(round(size3d[0] * 1.1))
        title_font_scale = int(round(size3d[0] * 1.8))

        # Tint the axis panes, RGB values from 0-1 and alpha denoting 
        # color intensity
        ax.w_xaxis.set_pane_color((0.9, 0.8, 0.9, 0.8))
        ax.w_yaxis.set_pane_color((0.8, 0.8, 0.9, 0.8))
        ax.w_zaxis.set_pane_color((0.9, 0.9, 0.8, 0.8))
        
        # Set z-axis to left hand side
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        
        # Set fontsize of axis ticks
        ax.tick_params(axis='both', which='major', labelsize=ax_font_scale, 
                       pad=10)
        
        # Label axes
        ax.set_xlabel(axis_label1, fontsize=ax_font_scale, 
                      labelpad=ax_font_scale*1.5)
        ax.set_ylabel(axis_label2, fontsize=ax_font_scale, 
                      labelpad=ax_font_scale*1.5)
        ax.set_zlabel(axis_label3, fontsize=ax_font_scale, 
                      labelpad=ax_font_scale*1.5)
 
        # Auto scale the z-axis
        ax.set_zlim(auto=True)
       
        # Set x-axis to decrease from left to right
        ax.invert_xaxis()
 
        # apply graph_scale so that if volatility is the x-axis it 
        # will be * 100
        ax.plot_surface(x,
                        y,
                        z,
                        rstride=2, cstride=2,
                        
                        # set the colormap to the chosen colorscheme
                        cmap=plt.get_cmap(colorscheme),
                        
                        # set the alpha value to the chosen 
                        # colorintensity
                        alpha=colorintensity,
                        linewidth=0.25)
       
        # Specify title 
        st = fig.suptitle(titlename, 
                          fontsize=title_font_scale, 
                          fontweight=0, 
                          color='black', 
                          style='italic', 
                          y=1.02)
 
        st.set_y(0.98)
        fig.subplots_adjust(top=1)
        
        if gif:
            return fig, ax, titlename, title_font_scale
        else:
            # Display graph
            plt.show()
    
    
    def _gif_defaults_setup(self, gif_folder=None, gif_filename=None, 
                            gif_type='2d'):
                
        if gif_folder is None:
            gif_folder = self.df_dict['df_gif_folder_'+gif_type]
        if gif_filename is None:
            gif_filename = self.df_dict['df_gif_filename_'+gif_type]
        
        working_folder = '{}/{}'.format(gif_folder, gif_filename)
        if not os.path.exists(working_folder):
            os.makedirs(working_folder)
        
        return gif_folder, gif_filename, working_folder
    
    
    def animated_3D_gif(
            self, S=None, r=None, q=None, sigma=None, option=None, 
            direction=None, notebook=None, colorscheme=None, 
            colorintensity=None, size3d=None, axis=None, spacegrain=None, 
            azim=None, elev=None, greek=None, num_sens=None, 
            gif_folder=None, gif_filename=None, gif_frame_update=None, 
            gif_min_dist=None, gif_max_dist=None, gif_min_elev=None, 
            gif_max_elev=None, gif_start_azim=None, gif_end_azim=None, 
            gif_dpi=None, gif_ms=None):
        """
        Create an animated gif of the selected greek 3D graph.

        Parameters
        ----------
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
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        notebook : Bool
            Whether the function is being run in an IPython notebook and 
            hence whether it should output in line or to an HTML file. 
            The default is False.
        colorscheme : Str
            The matplotlib colormap or plotly colorscale to use. The 
            default is 'jet' (which works in both).
        colorintensity : Float
            The alpha value indicating level of transparency / 
            intensity. The default is 1.
        size3d : Tuple
            Figure size for matplotlib chart. The default is (15, 12).
        axis : Str
            Whether the x-axis is 'price' or 'vol'. The default 
            is 'price'.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D 
            graphs. The default is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.
        greek : Str
            The sensitivity to be charted. Select from 'delta', 'gamma', 
            'vega', 'theta', 'rho', 'vomma', 'vanna', 'zomma', 'speed', 
            'color', 'ultima', 'vega_bleed', 'charm'. The default is 
            'delta'
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. 
            The default is False. 
        gif_folder : Str
            The folder to save the files into. The default is 'images/greeks'.
        gif_filename : Str
            The filename for the animated gif. The default is 'greek'.
        gif_frame_update : Int
            The number of degrees of rotation between each frame used to 
            construct the animated gif. The default is 2.
        gif_min_dist : Float
            The minimum zoom distance. The default is 9.0.
        gif_max_dist : Float
            The maximum zoom distance. The default is 10.0.
        gif_min_elev : Float
            The minimum elevation. The default is 10.0.
        gif_max_elev : Float
            The maximum elevation. The default is 60.0.
        gif_start_azim : Float
            The azimuth to start the gif from. The default is 0.
        gif_end_azim : Float
            The azimuth to end the gif on. The default is 360.
        gif_dpi : Int
            The image resolution to save. The default is 50 dpi.
        gif_ms : Int
            The time to spend on each frame in the gif. The default is 100ms.

        Returns
        -------
        Saves an animated gif.

        """
        gif=True
        
        gif_folder, gif_filename, working_folder = self._gif_defaults_setup(
            gif_folder=gif_folder, gif_filename=gif_filename, gif_type='3d')
        
        (gif_frame_update, gif_min_dist, gif_max_dist, gif_min_elev, 
         gif_max_elev, gif_start_azim, gif_end_azim, gif_dpi, 
         gif_ms) = itemgetter(
             'gif_frame_update', 'gif_min_dist', 'gif_max_dist', 
             'gif_min_elev', 'gif_max_elev', 'gif_start_azim', 'gif_end_azim', 
             'gif_dpi', 'gif_ms')(self._refresh_params_default(
                 gif_frame_update=gif_frame_update, gif_min_dist=gif_min_dist, 
                 gif_max_dist=gif_max_dist, gif_min_elev=gif_min_elev, 
                 gif_max_elev=gif_max_elev, gif_start_azim=gif_start_azim, 
                 gif_end_azim=gif_end_azim, gif_dpi=gif_dpi, gif_ms=gif_ms))    
                 
        fig, ax, titlename, title_font_scale = self.greeks_graphs_3D(
            S=S, r=r, q=q, sigma=sigma, option=option, 
            interactive=False, notebook=notebook, 
            colorscheme=colorscheme, colorintensity=colorintensity, 
            size3d=size3d, direction=direction, axis=axis, 
            spacegrain=spacegrain, azim=azim, elev=elev, greek=greek, 
            num_sens=num_sens, gif=gif)
        
        # number of degrees rotation between frames
        frame_update = gif_frame_update
        
        # set the range for horizontal rotation
        start_azim = gif_start_azim
        end_azim = gif_end_azim
        azim_range = end_azim - start_azim
                
        # set number of frames for the animated gif
        steps = math.floor(azim_range/frame_update)
        
        # a viewing perspective is composed of an elevation, distance, and 
        # azimuth define the range of values we'll cycle through for the 
        # distance of the viewing perspective
        min_dist = gif_min_dist
        max_dist = gif_max_dist
        dist_range = np.arange(min_dist, max_dist, (max_dist-min_dist)/steps)
        
        # define the range of values we'll cycle through for the elevation of 
        # the viewing perspective
        min_elev = gif_min_elev
        max_elev = gif_max_elev
        elev_range = np.arange(max_elev, min_elev, (min_elev-max_elev)/steps)
                
        # now create the individual frames that will be combined later into the 
        # animation
        for idx, azimuth in enumerate(
                range(start_azim, end_azim, frame_update)):
            
            # pan down, rotate around, and zoom out
            ax.azim = float(azimuth)
            ax.elev = elev_range[idx]
            ax.dist = dist_range[idx]

            # set the figure title
            st = fig.suptitle(titlename, 
                              fontsize=title_font_scale, 
                              fontweight=0, 
                              color='black', 
                              style='italic', 
                              y=1.02)
 
            st.set_y(0.98)
            fig.subplots_adjust(top=1)
            
            # save the image as a png file
            plt.savefig('{}/{}/img{:03d}.png'.format(gif_folder, gif_filename, 
                                                     azimuth), dpi=gif_dpi)
            
        # close the image object
        plt.close()
        
        # load all the static images into a list then save as an animated gif
        gif_filepath = '{}/{}.gif'.format(gif_folder, gif_filename)
        images = ([Image.open(image) for image in sorted(
            glob.glob('{}/*.png'.format(working_folder)))])
        gif = images[0]
        gif.info['duration'] = gif_ms #milliseconds per frame
        gif.info['loop'] = 0 #how many times to loop (0=infinite)
        gif.save(fp=gif_filepath, format='gif', save_all=True, 
                 append_images=images[1:])


    def animated_2D_gif(
            self, gif_folder=None, gif_filename=None, T=None, steps=None, 
            x_plot=None, y_plot=None, S=None, G1=None, G2=None, G3=None, 
            r=None, q=None, sigma=None, option=None, direction=None, 
            size2d=None, mpl_style=None, num_sens=None):
        """
        Create an animated gif of the selected pair of parameters.

        Parameters
        ----------
        gif_folder : Str
            The folder to save the files into. The default is 'images/greeks'.
        gif_filename : Str
            The filename for the animated gif. The default is 'greek'.
        T : Float
            Time to Maturity. The default is 0.5 (6 months).
        steps : Int
            Number of images to combine. The default is 40.
        x_plot : Str
                 The x-axis variable ('price', 'strike' or 'vol'). 
                 The default is 'price'.
        y_plot : Str
                 The y-axis variable ('value', 'delta', 'gamma', 'vega' 
                 or 'theta'). The default is 'delta.
        S : Float
             Underlying Stock Price. The default is 100.
        G1 : Float
             Strike Price of option 1. The default is 90.
        G2 : Float
             Strike Price of option 2. The default is 100.
        G3 : Float
             Strike Price of option 3. The default is 110.
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
        size2d : Tuple
            Figure size for matplotlib chart. The default is (6, 4).
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'. 
        num_sens : Bool
            Whether to calculate numerical or analytical sensitivity. 
            The default is False.

        Returns
        -------
        Saves an animated gif.

        """
        gif=True
        
        if T is None:
            T = 0.5
        if steps is None:
            steps = 40
        
        # Set up folders to save files 
        gif_folder, gif_filename, working_folder = self._gif_defaults_setup(
            gif_folder=gif_folder, gif_filename=gif_filename, gif_type='2d')
       
        # split the countdown from T to maturity in steps equal steps 
        time_steps = np.linspace(T, 0.001, steps)
        
        # create a plot for each time_step
        for counter, step in enumerate(time_steps):
            # create filenumber and filename
            filenumber = '{:03d}'.format(counter)
            filename = '{}, {}'.format(gif_filename, filenumber)
            
            # call the greeks_graphs_2d function to create graph
            fig, ax = self.greeks_graphs_2D(
                x_plot=x_plot, y_plot=y_plot, S=S, G1=G1, G2=G2, G3=G3, T=step, 
                T1=step, T2=step, T3=step, r=r, q=q, sigma=sigma, 
                option=option, direction=direction, size2d=size2d, 
                mpl_style=mpl_style, num_sens=num_sens, gif=gif)
            
            # save the image as a file 
            plt.savefig('{}/{}/img{}.png'.format(gif_folder, gif_filename, 
                                                     filename), dpi=50)
            # close the image object
            plt.close()
            
        # create a tuple of display durations, one for each frame
        #first_last = 100 #show the first and last frames for 100 ms
        #standard_duration = 10 #show all other frames for 10 ms
        #durations = tuple([first_last] + [standard_duration] * (
        #    len(time_steps) - 2) + [first_last])
        
        # load all the static images into a list then save as an animated gif
        gif_filepath = '{}/{}.gif'.format(gif_folder, gif_filename)
        images = ([Image.open(image) for image in sorted(
            glob.glob('{}/*.png'.format(working_folder)))])
        gif = images[0]
        gif.info['duration'] = 100#durations #ms per frame
        gif.info['loop'] = 0 #how many times to loop (0=infinite)
        gif.save(fp=gif_filepath, format='gif', save_all=True, 
                 append_images=images[1:])
    

    def _vis_greeks_3D(
            self, x, y, z, xmin, xmax, ymin, ymax, graph_scale, axis_label1, 
            axis_label2, direction, option, greek, interactive, colorscheme, 
            colorintensity, size3d, azim, elev, notebook, gif):
        """
        Display 3D greeks graph.

        Returns
        -------
        If 'interactive' is False, a matplotlib static graph.
        If 'interactive' is True, a plotly graph that can be rotated 
        and zoomed.

        """
        
        # Reverse the z-axis data if direction is 'short'
        if direction == 'short':
            z = -z
        
        titlename = self._titlename(option, direction, greek)
        axis_label3 = str(greek.title())

        if interactive:
        # Create a plotly graph    
            
            # Set the ranges for the contour values and reverse / rescale axes
            (x, y, x_start, x_stop, x_size, y_start, y_stop, y_size, z_start, 
             z_stop, z_size) = self._plotly_3D_ranges(
                 x, y, z, xmin, xmax, ymin, ymax, graph_scale)
            
            self._plotly_3D(x, y, z, x_start, x_stop, x_size, y_start, y_stop, 
                            y_size, z_start, z_stop, z_size, colorscheme, 
                            titlename, axis_label1, axis_label2, axis_label3, 
                            notebook)
    
        else:
        # Create a matplotlib graph
        
            x, y = self._mpl_axis_format(x, y, graph_scale)
        
            if gif:
                fig, ax, titlename, title_font_scale = self._mpl_3D(
                    x, y, z, size3d, azim, elev, axis_label1, axis_label2, 
                    axis_label3, colorscheme, colorintensity, titlename, gif)
                return fig, ax, titlename, title_font_scale
        
            else:
                self._mpl_3D(
                    x, y, z, size3d, azim, elev, axis_label1, axis_label2, 
                    axis_label3, colorscheme, colorintensity, titlename, gif)
       
    
    def payoffs(self, S=None, K=None, K1=None, K2=None, K3=None, K4=None, 
                T=None, r=None, q=None, sigma=None, option=None, 
                direction=None, size2d=None, cash=None, ratio=None, value=None, 
                combo_payoff=None, mpl_style=None):
        """
        Displays the graph of the specified combo payoff.
                
        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        K : Float
            Strike Price of option 1. The default is 100 
            (individual payoffs may have own defaults).
        K1 : Float
             Strike Price of option 1. The default is 95 
             (individual payoffs may have own defaults).
        K2 : Float
             Strike Price of option 2. The default is 105 
             (individual payoffs may have own defaults).
        K3 : Float
             Strike Price of option 3. The default is 105 
             (individual payoffs may have own defaults).
        K4 : Float
             Strike Price of option 4. The default is 105 
             (individual payoffs may have own defaults).
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
            Whether to discount forward to present value. The default 
            is False.
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. The 
            default is 2. 
        value : Bool
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        combo_payoff : Str
            The payoff to be displayed.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        Runs the specified combo payoff method.

        """
        if combo_payoff is None:
            combo_payoff = self.df_dict['df_combo_payoff']
        
        if combo_payoff == 'call':
            self.call(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                      direction=direction, value=value, 
                      mpl_style=mpl_style, size2d=size2d)
        
        if combo_payoff == 'put':
            self.put(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                     direction=direction, value=value, mpl_style=mpl_style, 
                     size2d=size2d)
        
        if combo_payoff == 'stock':
            self.stock(S=S, direction=direction, mpl_style=mpl_style, 
                       size2d=size2d)
        
        if combo_payoff == 'forward':
            self.forward(S=S, T=T, r=r, q=q, sigma=sigma, 
                         direction=direction, cash=cash, mpl_style=mpl_style, 
                         size2d=size2d)
        
        if combo_payoff == 'collar':
            self.collar(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                        direction=direction, value=value, mpl_style=mpl_style, 
                        size2d=size2d)
        
        if combo_payoff == 'spread':
            self.spread(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                        option=option, direction=direction, value=value, 
                        mpl_style=mpl_style, size2d=size2d)
            
        if combo_payoff == 'backspread':
            self.backspread(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                            option=option, ratio=ratio, value=value, 
                            mpl_style=mpl_style, size2d=size2d)
        
        if combo_payoff == 'ratio vertical spread':
            self.ratio_vertical_spread(
                S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option, 
                ratio=ratio, value=value, mpl_style=mpl_style, size2d=size2d)
        
        if combo_payoff == 'straddle':
            self.straddle(S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                          direction=direction, value=value, 
                          mpl_style=mpl_style, size2d=size2d)

        if combo_payoff == 'strangle':
            self.strangle(S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                          direction=direction, value=value, 
                          mpl_style=mpl_style, size2d=size2d)
        
        if combo_payoff == 'butterfly':    
            self.butterfly(S=S, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, 
                           sigma=sigma, option=option, direction=direction, 
                           value=value, mpl_style=mpl_style, size2d=size2d)
        
        if combo_payoff == 'christmas tree':
            self.christmas_tree(
                S=S, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                option=option, direction=direction, value=value, 
                mpl_style=mpl_style, size2d=size2d)    

        if combo_payoff == 'condor':
            self.condor(S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, 
                        sigma=sigma, option=option, direction=direction, 
                        value=value, mpl_style=mpl_style, size2d=size2d)
        
        if combo_payoff == 'iron butterfly':
            self.iron_butterfly(
                S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, sigma=sigma, 
                direction=direction, value=value, mpl_style=mpl_style, 
                size2d=size2d)
            
        if combo_payoff == 'iron condor':
            self.iron_condor(
                S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, sigma=sigma, 
                direction=direction, value=value, mpl_style=mpl_style, 
                size2d=size2d)
            
    
    def call(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
             direction=None, value=None, mpl_style=None, size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'call'
                
        # Update pricing input parameters to default if not supplied
        (S, K, T, r, q, sigma, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K', 'T', 'r', 'q', 'sigma', 'direction', 'value',
            'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                value=value, mpl_style=mpl_style, size2d=size2d))
        
        # Calculate option prices
        SA, C1_0, C1, C1_G = self._return_options(
            legs=1, S=S, K1=K, T1=T, r=r, q=q, sigma=sigma, option1='call')
        
        # Create payoff based on direction
        if direction == 'long':
            payoff = C1 - C1_0
            title = 'Long Call'
            if value:
                payoff2 = C1_G - C1_0
            else:
                payoff2 = None

        if direction == 'short':
            payoff = -C1 + C1_0
            title = 'Short Call'
            if value:
                payoff2 = -C1_G + C1_0
            else:
                payoff2 = None
        
        # Visualize payoff        
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)   
                
        
    def put(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            direction=None, value=None, mpl_style=None, size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'. 

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'put'
                
        # Update pricing input parameters to default if not supplied
        (S, K, T, r, q, sigma, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K', 'T', 'r', 'q', 'sigma', 'direction', 'value',
            'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                value=value, mpl_style=mpl_style, size2d=size2d))
        
        # Calculate option prices
        SA, C1_0, C1, C1_G = self._return_options(
            legs=1, S=S, K1=K, T1=T, r=r, q=q, sigma=sigma, option1='put')
        
        # Create payoff based on direction
        if direction == 'long':
            payoff = C1 - C1_0
            title = 'Long Put'
            if value:
                payoff2 = C1_G - C1_0
            else:
                payoff2 = None

        if direction == 'short':
            payoff = -C1 + C1_0
            title = 'Short Put'
            if value:
                payoff2 = -C1_G + C1_0
            else:
                payoff2 = None
        
        # Visualize payoff        
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)   
               
        
    def stock(self, S=None, direction=None, mpl_style=None, size2d=None):
        """
        Displays the graph of the underlying.

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100. 
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'. 

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'stock'
        
        # Update pricing input parameters to default if not supplied
        S, direction, mpl_style, size2d = itemgetter(
            'S', 'direction', 'mpl_style', 
            'size2d')(self._refresh_combo_params_default(
                S=S, direction=direction, mpl_style=mpl_style, size2d=size2d))
        
        # Define strike range
        SA = np.linspace(0.75 * S, 1.25 * S, 1000)
        
        # Create payoff based on option type
        if direction == 'long':
            payoff = SA - S
            title = 'Long Stock'
        
        if direction == 'short':
            payoff = S - SA
            title = 'Short Stock'
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=None, size2d=size2d, mpl_style=mpl_style)     
            
    
    def forward(self, S=None, T=None, r=None, q=None, sigma=None, 
                direction=None, cash=None, mpl_style=None, size2d=None):
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
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'forward'
        
        # Update pricing input parameters to default if not supplied
        (S, T, r, q, sigma, direction, cash, mpl_style, 
         size2d) = itemgetter(
            'S', 'T', 'r', 'q', 'sigma', 'direction', 'cash',
            'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                cash=cash, mpl_style=mpl_style, size2d=size2d))
                   
        # Calculate option prices
        SA, C1_0, C1, C1_G, C2_0, C2, C2_G = self._return_options(
            legs=2, S=S, K1=S, T1=T, r=r, q=q, sigma=sigma, option1='call', 
            K2=S, T2=T, option2='put')
        
        # Whether to discount the payoff
        if cash:
            pv = np.exp(-r * T)
        else:    
            pv = 1
               
        # Create payoff based on option type
        if direction == 'long':
            payoff = (C1 - C2 - C1_0 + C2_0) * pv
            title = 'Long Forward'
            
        if direction == 'short':
            payoff = -C1 + C2 + C1_0 - C2_0 * pv
            title = 'Short Forward'
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         size2d=size2d, mpl_style=mpl_style, payoff2=None)
    
    
    def collar(self, S=None, K1=None, K2=None, T=None, r=None, q=None, 
               sigma=None, direction=None, value=None, mpl_style=None, 
               size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'collar'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, T, r, q, sigma, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'T', 'r', 'q', 'sigma', 'direction', 'value',
            'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                direction=direction, value=value, mpl_style=mpl_style, 
                size2d=size2d))
   
        # Calculate option prices
        SA, C1_0, C1, C1_G, C2_0, C2, C2_G = self._return_options(
            legs=2, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1='put', 
            K2=K2, T2=T, option2='call')
        
        # Create payoff based on option type
        if direction == 'long':
            payoff = (SA - S + C1 - C2 - C1_0 + C2_0)
            title = 'Long Collar'
            if value:
                payoff2 = (SA - S + C1_G - C2_G - C1_0 + C2_0)
            else:
                payoff2 = None
                
        if direction == 'short':
            payoff = (-SA + S - C1 + C2 + C1_0 - C2_0)
            title = 'Short Collar'
            if value:
                payoff2 = (-SA + S - C1_G + C2_G + C1_0 - C2_0)
            else:
                payoff2 = None
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)

    
    
    def spread(self, S=None, K1=None, K2=None, T=None, r=None, q=None, 
               sigma=None, option=None, direction=None, value=None, 
               mpl_style=None, size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.


        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'spread'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, T, r, q, sigma, option, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'T', 'r', 'q', 'sigma', 'option', 'direction', 
            'value', 'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                direction=direction, value=value, mpl_style=mpl_style, 
                size2d=size2d))
               
        # Calculate option prices
        SA, C1_0, C1, C1_G, C2_0, C2, C2_G = self._return_options(
            legs=2, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1=option, 
            K2=K2, T2=T, option2=option)
 
        # Create payoff based on option type
        if direction == 'long':        
            payoff = C1 - C2 - C1_0 + C2_0
            if value:
                payoff2 = C1_G - C2_G - C1_0 + C2_0
            else:
                payoff2 = None
                
        if direction == 'short':
            payoff = -C1 + C2 + C1_0 - C2_0
            if value:
                payoff2 = -C1_G + C2_G + C1_0 - C2_0
            else:
                payoff2 = None
        
        # Create title based on option type and direction       
        if option == 'call' and direction == 'long':
            title = 'Bull Call Spread'
        if option == 'put' and direction == 'long':
            title = 'Bull Put Spread'
        if option == 'call' and direction == 'short':
            title = 'Bear Call Spread'
        if option == 'put' and direction == 'short':
            title = 'Bear Put Spread' 
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)
        
   
    def backspread(self, S=None, K1=None, K2=None, T=None, r=None, q=None, 
                   sigma=None, option=None, ratio=None, value=None, 
                   mpl_style=None, size2d=None):
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
            Multiple of OTM options to be sold for ITM purchased. The 
            default is 2.    
        value : Bool
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'backspread'

        # Update pricing input parameters to default if not supplied
        (S, K1, K2, T, r, q, sigma, option, ratio, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'T', 'r', 'q', 'sigma', 'option', 'ratio', 
            'value', 'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                ratio=ratio, value=value, mpl_style=mpl_style, size2d=size2d))
        
        # Calculate option prices
        SA, C1_0, C1, C1_G, C2_0, C2, C2_G = self._return_options(
            legs=2, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1=option, 
            K2=K2, T2=T, option2=option)
        
        # Create payoff based on option type
        if option == 'call':
            title = 'Call Backspread'
            payoff = (-C1 + (ratio * C2) + C1_0 - (ratio * C2_0))
            if value:
                payoff2 = (-C1_G + (ratio * C2_G) + C1_0 - (ratio * C2_0))
            else:
                payoff2 = None
        
        if option == 'put':
            payoff = (ratio * C1 - C2 - ratio * C1_0 + C2_0)
            title = 'Put Backspread'
            if value:
                payoff2 = (ratio * C1_G - C2_G - ratio * C1_0 + C2_0)
            else:
                payoff2 = None
        
        # Visualize payoff        
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)
        
        
    def ratio_vertical_spread(self, S=None, K1=None, K2=None, T=None, r=None, 
                              q=None, sigma=None, option=None, ratio=None, 
                              value=None, mpl_style=None, size2d=None):
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
            Multiple of OTM options to be sold for ITM purchased. The 
            default is 2.    
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.
        
        """
        
        # Specify the combo payoff so that parameter initialisation takes 
        # into account specific defaults
        self.combo_payoff = 'ratio vertical spread'

        # Update pricing input parameters to default if not supplied
        (S, K1, K2, T, r, q, sigma, option, ratio, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'T', 'r', 'q', 'sigma', 'option', 'ratio', 
            'value', 'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, option=option,
                ratio=ratio, value=value, mpl_style=mpl_style, size2d=size2d))
             
        # Calculate option prices
        SA, C1_0, C1, C1_G, C2_0, C2, C2_G = self._return_options(
            legs=2, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1=option, 
            K2=K2, T2=T, option2=option)
        
        # Create payoff based on option type
        if option == 'call':
            title = 'Call Ratio Vertical Spread'
            payoff = (C1 - ratio * C2 - C1_0 + ratio * C2_0)
            if value:
                payoff2 = (C1_G - ratio * C2_G - C1_0 + ratio * C2_0)
            else:
                payoff2 = None

        if option == 'put':
            title = 'Put Ratio Vertical Spread'
            payoff = (-ratio * C1 + C2 + ratio * C1_0 - C2_0)
            if value:
                payoff2 = (-ratio * C1_G + C2_G + ratio * C1_0 - C2_0)
            else:
                payoff2 = None
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)
        
    
    def straddle(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                 direction=None, value=None, mpl_style=None, size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     
        
        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.    

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'straddle'
        
        # Update pricing input parameters to default if not supplied
        (S, K, T, r, q, sigma, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K', 'T', 'r', 'q', 'sigma', 'direction', 'value', 
            'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K=K, T=T, r=r, q=q, sigma=sigma, direction=direction, 
                value=value, mpl_style=mpl_style, size2d=size2d))
                
        # Calculate option prices
        SA, C1_0, C1, C1_G, C2_0, C2, C2_G = self._return_options(
            legs=2, S=S, K1=K, T1=T, r=r, q=q, sigma=sigma, option1='put', 
            K2=K, T2=T, option2='call')
        
        # Create payoff based on direction
        if direction == 'long':
            payoff = C1 + C2 - C1_0 - C2_0
            title = 'Long Straddle'
            if value:
                payoff2 = C1_G + C2_G - C1_0 - C2_0
            else:
                payoff2 = None
                        
        if direction == 'short':
            payoff = -C1 - C2 + C1_0 + C2_0
            title = 'Short Straddle'
            if value:
                payoff2 = -C1_G - C2_G + C1_0 + C2_0
            else:
                payoff2 = None
        
        # Visualize payoff    
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)    
  
    
    def strangle(self, S=None, K1=None, K2=None, T=None, r=None, q=None, 
                 sigma=None, direction=None, value=None, mpl_style=None, 
                 size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.
        
        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'strangle'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, T, r, q, sigma, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'T', 'r', 'q', 'sigma', 'direction', 'value', 
            'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, T=T, r=r, q=q, sigma=sigma, 
                direction=direction, value=value, mpl_style=mpl_style, 
                size2d=size2d))
                
        # Calculate option prices
        SA, C1_0, C1, C1_G, C2_0, C2, C2_G = self._return_options(
            legs=2, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1='put', 
            K2=K2, T2=T, option2='call')
        
        # Create payoff based on direction
        if direction == 'long':
            payoff = C1 + C2 - C1_0 - C2_0
            title = 'Long Strangle'
            if value:
                payoff2 = C1_G + C2_G - C1_0 - C2_0
            else:
                payoff2 = None
        
        if direction == 'short':
            payoff = -C1 - C2 + C1_0 + C2_0
            title = 'Short Strangle'
            if value:
                payoff2 = -C1_G - C2_G + C1_0 + C2_0
            else:
                payoff2 = None
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)    


    def butterfly(self, S=None, K1=None, K2=None, K3=None, T=None, r=None, 
                  q=None, sigma=None, option=None, direction=None, value=None, 
                  mpl_style=None, size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'butterfly'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, K3, T, r, q, sigma, option, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'K3', 'T', 'r', 'q', 'sigma', 'option', 
            'direction', 'value', 'mpl_style', 
            'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                option=option, direction=direction, value=value, 
                mpl_style=mpl_style, size2d=size2d))
                
        # Calculate option prices
        (SA, C1_0, C1, C1_G, C2_0, C2, C2_G, C3_0, C3, 
         C3_G) = self._return_options(
             legs=3, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1=option, 
             K2=K2, T2=T, option2=option, K3=K3, T3=T, option3=option)
        
        # Create payoff based on direction
        if direction == 'long':
            payoff = (C1 - 2 * C2 + C3 - C1_0 + 2 * C2_0 - C3_0)
            if value:
                payoff2 = (C1_G - 2 * C2_G + C3_G - C1_0 + 2 * C2_0 - C3_0)
            else:
                payoff2 = None
                
        if direction == 'short':    
            payoff = (-C1 + 2 * C2 - C3 + C1_0 - 2 * C2_0 + C3_0)
            if value:
                payoff2 = (-C1_G + 2 * C2_G - C3_G + C1_0 - 2 * C2_0 + C3_0)
            else:
                payoff2 = None
        
        # Create title based on option type and direction                 
        if option == 'call' and direction == 'long':
            title = 'Long Butterfly with Calls'
        if option == 'put' and direction == 'long':
            title = 'Long Butterfly with Puts'
        if option == 'call' and direction == 'short':
            title = 'Short Butterfly with Calls'
        if option == 'put' and direction == 'short':
            title = 'Short Butterfly with Puts'
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)

    
    def christmas_tree(self, S=None, K1=None, K2=None, K3=None, T=None, r=None, 
                       q=None, sigma=None, option=None, direction=None, 
                       value=None, mpl_style=None, size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.
        
        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'christmas tree'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, K3, T, r, q, sigma, option, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'K3', 'T', 'r', 'q', 'sigma', 'option', 
            'direction', 'value', 'mpl_style', 
            'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, K3=K3, T=T, r=r, q=q, sigma=sigma, 
                option=option, direction=direction, value=value, 
                mpl_style=mpl_style, size2d=size2d))
                 
        # Calculate option prices
        (SA, C1_0, C1, C1_G, C2_0, C2, C2_G, C3_0, C3, 
         C3_G) = self._return_options(
             legs=3, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1=option, 
             K2=K2, T2=T, option2=option, K3=K3, T3=T, option3=option)
        
        # Create payoff based on option type and direction
        if option == 'call' and direction == 'long':
            payoff = (C1 - C2 - C3 - C1_0 + C2_0 + C3_0)
            title = 'Long Christmas Tree with Calls'
            if value:
                payoff2 = (C1_G - C2_G - C3_G - C1_0 + C2_0 + C3_0)
            else:
                payoff2 = None
                
        if option == 'put' and direction == 'long':
            payoff = (-C1 - C2 + C3 + C1_0 + C2_0 - C3_0)
            title = 'Long Christmas Tree with Puts'
            if value:
                payoff2 = (-C1_G - C2_G + C3_G + C1_0 + C2_0 - C3_0)
            else:
                payoff2 = None
            
        if option == 'call' and direction == 'short':
            payoff = (-C1 + C2 + C3 + C1_0 - C2_0 - C3_0)
            title = 'Short Christmas Tree with Calls'
            if value:
                payoff2 = (-C1_G + C2_G + C3_G + C1_0 - C2_0 - C3_0)
            else:
                payoff2 = None
            
        if option == 'put' and direction == 'short':
            payoff = (C1 + C2 - C3 - C1_0 - C2_0 + C3_0)
            title = 'Short Christmas Tree with Puts'
            if value:
                payoff2 = (C1_G + C2_G - C3_G - C1_0 - C2_0 + C3_0)
            else:
                payoff2 = None
        
        # Visualize payoff    
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)


    def condor(self, S=None, K1=None, K2=None, K3=None, K4=None, T=None, 
               r=None, q=None, sigma=None, option=None, direction=None, 
               value=None, mpl_style=None, size2d=None):
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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults        
        self.combo_payoff = 'condor'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, K3, K4, T, r, q, sigma, option, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'K3', 'K4', 'T', 'r', 'q', 'sigma', 'option', 
            'direction', 'value', 'mpl_style', 
            'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, sigma=sigma, 
                option=option, direction=direction, value=value, 
                mpl_style=mpl_style, size2d=size2d))
                
        # Calculate option prices
        (SA, C1_0, C1, C1_G, C2_0, C2, C2_G, C3_0, C3, C3_G, C4_0, C4, 
         C4_G) = self._return_options(
            legs=4, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1=option, 
            K2=K2, T2=T, option2=option, K3=K3, T3=T, option3=option, K4=K4, 
            T4=T, option4=option)
        
        # Create payoff based on direction
        if direction == 'long':
            payoff = (C1 - C2 - C3 + C4 - C1_0 + C2_0 + C3_0 - C4_0)
            if value:
                payoff2 = (C1_G - C2_G - C3_G + C4_G 
                           - C1_0 + C2_0 + C3_0 - C4_0)
            else:
                payoff2 = None
        
        if direction == 'short':
            payoff = (-C1 + C2 + C3 - C4 + C1_0 - C2_0 - C3_0 + C4_0)
            if value:
                payoff2 = (-C1_G + C2_G + C3_G - C4_G 
                           + C1_0 - C2_0 - C3_0 + C4_0)
            else:
                payoff2 = None
        
        # Create title based on option type and direction        
        if option == 'call' and direction == 'long':
            title = 'Long Condor with Calls'
        if option == 'put' and direction == 'long':
            title = 'Long Condor with Puts'
        if option == 'call' and direction == 'short':
            title = 'Short Condor with Calls'
        if option == 'put' and direction == 'short':
            title = 'Short Condor with Puts'    
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)


    def iron_butterfly(self, S=None, K1=None, K2=None, K3=None, K4=None, 
                       T=None, r=None, q=None, sigma=None, direction=None, 
                       value=None, mpl_style=None, size2d=None):
        """
        Displays the graph of the iron butterfly strategy:
            Short one OTM put
            Long one ATM put
            Long one ATM call 
            Short one OTM call
        Akin to having a long straddle inside a larger short strangle 
        (or vice-versa)

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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'iron butterfly'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, K3, K4, T, r, q, sigma, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'K3', 'K4', 'T', 'r', 'q', 'sigma', 'direction', 
            'value', 'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, sigma=sigma, 
                direction=direction, value=value, mpl_style=mpl_style, 
                size2d=size2d))
           
        # Calculate option prices
        (SA, C1_0, C1, C1_G, C2_0, C2, C2_G, C3_0, C3, C3_G, C4_0, C4, 
         C4_G) = self._return_options(
            legs=4, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1='put', 
            K2=K2, T2=T, option2='call', K3=K3, T3=T, option3='put', K4=K4, 
            T4=T, option4='call')
        
        # Create payoff based on direction
        if direction == 'long':
            payoff = (-C1 + C2 + C3 - C4 + C1_0 - C2_0 - C3_0 + C4_0)
            title = 'Long Iron Butterfly'
            if value:
                payoff2 = (-C1_G + C2_G + C3_G - C4_G 
                           + C1_0 - C2_0 - C3_0 + C4_0)
            else:
                payoff2 = None
        
        if direction == 'short':
            payoff = (C1 - C2 - C3 + C4 - C1_0 + C2_0 + C3_0 - C4_0)
            title = 'Short Iron Butterfly'
            if value:
                payoff2 = (C1_G - C2_G - C3_G + C4_G 
                           - C1_0 + C2_0 + C3_0 - C4_0)
            else:
                payoff2 = None
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)
    
    
    def iron_condor(self, S=None, K1=None, K2=None, K3=None, K4=None, T=None, 
                    r=None, q=None, sigma=None, direction=None, value=None, 
                    mpl_style=None, size2d=None):
        """
        Displays the graph of the iron condor strategy:
            Long one OTM put
            Short one OTM put with a higher strike
            Short one OTM call 
            Long one OTM call with a higher strike
        Akin to having a long strangle inside a larger short strangle 
        (or vice-versa)   

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
            Whether to show the current value as well as the terminal 
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs. 
            The default is 'seaborn-darkgrid'.     

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """
        
        # Specify the combo payoff so that parameter initialisation 
        # takes into account specific defaults
        self.combo_payoff = 'iron condor'
        
        # Update pricing input parameters to default if not supplied
        (S, K1, K2, K3, K4, T, r, q, sigma, direction, value, mpl_style, 
         size2d) = itemgetter(
            'S', 'K1', 'K2', 'K3', 'K4', 'T', 'r', 'q', 'sigma', 'direction', 
            'value', 'mpl_style', 'size2d')(self._refresh_combo_params_default(
                S=S, K1=K1, K2=K2, K3=K3, K4=K4, T=T, r=r, q=q, sigma=sigma, 
                direction=direction, value=value, mpl_style=mpl_style, 
                size2d=size2d))
        
        # Calculate option prices
        (SA, C1_0, C1, C1_G, C2_0, C2, C2_G, C3_0, C3, C3_G, C4_0, C4, 
         C4_G) = self._return_options(
            legs=4, S=S, K1=K1, T1=T, r=r, q=q, sigma=sigma, option1='put', 
            K2=K2, T2=T, option2='put', K3=K3, T3=T, option3='call', K4=K4, 
            T4=T, option4='call')
        
        # Create payoff based on direction and value flag
        if direction == 'long':
            payoff = (C1 - C2 - C3 + C4 - C1_0 + C2_0 + C3_0 - C4_0)
            if value:
                payoff2 = (C1_G - C2_G - C3_G + C4_G 
                           - C1_0 + C2_0 + C3_0 - C4_0)
            else:
                payoff2 = None
        
        elif direction == 'short':
            payoff = (-C1 + C2 + C3 - C4 + C1_0 - C2_0 - C3_0 + C4_0)
            if value:
                payoff2 = (-C1_G + C2_G + C3_G - C4_G 
                           + C1_0 - C2_0 - C3_0 + C4_0)
            else:
                payoff2 = None
              
        # Create graph title based on direction 
        if direction == 'long':
            title = 'Long Iron Condor'
        
        if direction == 'short':
            title = 'Short Iron Condor'
        
        # Visualize payoff
        self._vis_payoff(S=S, SA=SA, payoff=payoff, title=title, 
                         payoff2=payoff2, size2d=size2d, mpl_style=mpl_style)

    
    def _vis_payoff(self, S, SA, payoff, title, payoff2, size2d, mpl_style):
        """
        Display the payoff diagrams using matplotlib

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        SA : Array
             Range of Strikes to provide x-axis values. The default 
             is 75% to 125%.
        payoff : Array
            Terminal payoff value less initial cost.
        label : Str
            Label for terminal payoff. The default is 'Payoff'
        title : Str
            Chart title giving name of combo. The default 
            is 'Option Payoff'
        payoff2 : Array
            Current payoff value less initial cost.
        label2 : Str
            Label for current payoff value.
       
        Returns
        -------
        2D Payoff Graph.

        """
        
        # Use seaborn darkgrid style 
        plt.style.use(mpl_style)
        
        # Update chart parameters
        pylab.rcParams.update(self.mpl_params)
        
        # Create the figure and axes objects
        fig = plt.figure(figsize=size2d)
        
        # Use gridspec to allow modification of bounding box
        gs1 = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs1[0])
        
        # Plot the terminal payoff
        ax.plot(SA, payoff, color='blue', label='Payoff')
        
        # If the value flag is selected, plot the payoff with the 
        # current time to maturity
        if payoff2 is not None:
            ax.plot(SA, payoff2, color='red', label='Value')
        
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
        ax.set(xlabel='Stock Price', ylabel='P&L', title=title)
        
        # Create a legend
        ax.legend(loc=0, fontsize=10)
        
        # Apply tight layout
        gs1.tight_layout(fig, rect=[0, 0, 1, 1])
              
        # Display the chart
        plt.show()
    
    
    def _return_options(
            self, legs, S, K1, T1, r, q, sigma, option1, K2=None, 
            T2=None, option2=None, K3=None, T3=None, option3=None, K4=None, 
            T4=None, option4=None):
        """
        Calculate option prices to be used in payoff diagrams.

        Parameters
        ----------
        legs : Int
            Number of option legs to calculate. 

        Returns
        -------
        From 1 to 4 sets of option values:
            Cx_0: Current option price; Float.
            Cx: Terminal Option payoff, varying by strike; Array
            Cx_G: Current option value, varying by strike; Array

        """
        
        # create array of 1000 equally spaced points between 75% of 
        # initial underlying price and 125%
        SA = np.linspace(0.75 * S, 1.25 * S, 1000)
        
        # Calculate the current price of option 1       
        C1_0 = self.price(S=S, K=K1, T=T1, r=r, q=q, sigma=sigma, 
                          option=option1, default=False)
        
        # Calculate the prices at maturity for the range of strikes 
        # in SA of option 1
        C1 = self.price(S=SA, K=K1, T=0, r=r, q=q, sigma=sigma, 
                        option=option1, default=False)
        
        # Calculate the current prices for the range of strikes 
        # in SA of option 1
        C1_G = self.price(S=SA, K=K1, T=T1, r=r, q=q, sigma=sigma, 
                          option=option1, default=False)
        
        
        if legs > 1:
            # Calculate the current price of option 2
            C2_0 = self.price(S=S, K=K2, T=T2, r=r, q=q, sigma=sigma, 
                              option=option2, default=False)
            
            # Calculate the prices at maturity for the range of strikes 
            # in SA of option 2
            C2 = self.price(S=SA, K=K2, T=0, r=r, q=q, sigma=sigma, 
                            option=option2, default=False)
            
            # Calculate the current prices for the range of strikes 
            # in SA of option 2
            C2_G = self.price(S=SA, K=K2, T=T2, r=r, q=q, sigma=sigma, 
                              option=option2, default=False)

        if legs > 2:
            # Calculate the current price of option 3
            C3_0 = self.price(S=S, K=K3, T=T3, r=r, q=q, sigma=sigma, 
                              option=option3, default=False)
            
            # Calculate the prices at maturity for the range of strikes 
            # in SA of option 3
            C3 = self.price(S=SA, K=K3, T=0, r=r, q=q, sigma=sigma, 
                            option=option3, default=False)
            
            # Calculate the current prices for the range of strikes 
            # in SA of option 3
            C3_G = self.price(S=SA, K=K3, T=T3, r=r, q=q, sigma=sigma, 
                              option=option3, default=False)
        
        if legs > 3:
            # Calculate the current price of option 4
            C4_0 = self.price(S=S, K=K4, T=T4, r=r, q=q, sigma=sigma, 
                              option=option4, default=False)
            
            # Calculate the prices at maturity for the range of strikes 
            # in SA of option 4
            C4 = self.price(S=SA, K=K4, T=0, r=r, q=q, sigma=sigma, 
                            option=option4, default=False)
            
            # Calculate the current prices for the range of strikes 
            # in SA of option 4
            C4_G = self.price(S=SA, K=K4, T=T4, r=r, q=q, sigma=sigma, 
                              option=option4, default=False)
       
        if legs == 1:
            return SA, C1_0, C1, C1_G
        
        if legs == 2:
            return SA, C1_0, C1, C1_G, C2_0, C2, C2_G
        
        if legs == 3:
            return SA, C1_0, C1, C1_G, C2_0, C2, C2_G, C3_0, C3, C3_G
        
        if legs == 4:
            return (SA, C1_0, C1, C1_G, C2_0, C2, C2_G, C3_0, C3, C3_G, C4_0, 
                    C4, C4_G)
        
        
    
